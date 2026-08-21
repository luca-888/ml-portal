(() => {
  const roots = document.querySelectorAll('[data-guide="llm-c"]');
  roots.forEach((root) => initLab(root));

  function initLab(root) {
    const lab = root.querySelector("[data-lc-lab]");
    if (!lab) return;

    const $ = (selector) => lab.querySelector(selector);
    const $$ = (selector) => [...lab.querySelectorAll(selector)];
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const VP = 50304;
    let activePhase = "forward";
    let lastResult;

    lab.dataset.reducedMotion = reducedMotion ? "true" : "false";

    const clamp = (value, min, max) => Math.min(max, Math.max(min, Number(value) || min));
    const formatCount = (value) => {
      if (value >= 1e9) return `${(value / 1e9).toFixed(value >= 1e10 ? 1 : 2)}B`;
      if (value >= 1e6) return `${(value / 1e6).toFixed(value >= 1e8 ? 1 : 2)}M`;
      if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
      return Math.round(value).toLocaleString("zh-CN");
    };
    const formatBytes = (value) => {
      const units = ["B", "KiB", "MiB", "GiB", "TiB"];
      let amount = Math.max(0, value);
      let unit = 0;
      while (amount >= 1024 && unit < units.length - 1) {
        amount /= 1024;
        unit += 1;
      }
      const digits = amount >= 100 ? 0 : amount >= 10 ? 1 : 2;
      return `${amount.toFixed(digits)} ${units[unit]}`;
    };
    const formatPercent = (value) => `${value.toFixed(value < 10 ? 1 : 0)}%`;

    function readConfig() {
      const values = {};
      for (const [key, min, max] of [
        ["b", 1, 64],
        ["t", 16, 2048],
        ["c", 64, 4096],
        ["l", 1, 96],
      ]) {
        const input = $(`[data-lc-input="${key}"]`);
        const value = clamp(input.value, min, max);
        input.value = value;
        values[key] = value;
      }
      values.dtype = $('[data-lc-input="dtype"]').value === "fp32" ? "fp32" : "bf16";
      values.path = $('input[name="lc-path"]:checked')?.value === "cpu" ? "cpu" : "cuda";
      values.nh = Math.max(1, Math.round(values.c / 64));
      values.maxT = Math.max(1024, values.t);
      values.effectiveDtype = values.path === "cpu" ? "fp32" : values.dtype;
      values.elementBytes = values.effectiveDtype === "bf16" ? 2 : 4;
      return values;
    }

    function parameterCount({ c: C, l: L, maxT }) {
      return VP * C + maxT * C + L * (12 * C * C + 13 * C) + 2 * C;
    }

    function cudaActivationBytes(config) {
      const { b: B, t: T, c: C, l: L, nh: NH, elementBytes } = config;
      const btc = B * T * C;
      const mainElements = (7 + 16 * L) * btc;
      const attentionElements = L * B * NH * T * T;
      const outputElements = B * T * Math.max(3 * C, NH * T, VP);
      const fp32StatsElements = (4 * L + 3) * B * T;
      return {
        main: mainElements * elementBytes,
        attention: attentionElements * elementBytes,
        output: outputElements * elementBytes,
        stats: fp32StatsElements * 4,
        total: (mainElements + attentionElements + outputElements) * elementBytes + fp32StatsElements * 4,
      };
    }

    function cpuActivationBytes(config) {
      const { b: B, t: T, c: C, l: L, nh: NH } = config;
      const btc = B * T * C;
      const mainElements = (2 + 18 * L) * btc;
      const attentionElements = 2 * L * B * NH * T * T;
      const logitsElements = 2 * B * T * VP;
      const statsElements = (4 * L + 3) * B * T;
      const forward = (mainElements + attentionElements + logitsElements + statsElements) * 4;
      return {
        main: mainElements * 4,
        attention: attentionElements * 4,
        output: logitsElements * 4,
        stats: statsElements * 4,
        total: forward,
        gradients: forward,
      };
    }

    function estimate(config) {
      const params = parameterCount(config);
      const parameterBytes = params * config.elementBytes;
      const gradientBytes = params * config.elementBytes;
      const optimizerBytes = params * 8;
      const masterBytes = config.path === "cuda" && config.effectiveDtype === "bf16" ? params * 4 : 0;
      const acts = config.path === "cuda" ? cudaActivationBytes(config) : cpuActivationBytes(config);
      const inputBytes = config.b * config.t * 2 * 4;
      const activationGradientBytes = config.path === "cpu" ? acts.gradients : 0;

      const buffers = [
        { id: "params", label: "Parameters", detail: `${config.effectiveDtype.toUpperCase()} 参数`, bytes: parameterBytes },
        { id: "grads", label: "Parameter gradients", detail: `${config.effectiveDtype.toUpperCase()} 参数梯度`, bytes: gradientBytes },
        { id: "optim", label: "AdamW m + v", detail: "两个 FP32 状态", bytes: optimizerBytes },
        ...(masterBytes ? [{ id: "master", label: "Master weights", detail: "FP32 参数副本", bytes: masterBytes }] : []),
        { id: "acts", label: "Forward activations", detail: config.path === "cpu" ? "23 个张量分区" : "21 个张量与 scratch", bytes: acts.total },
        ...(activationGradientBytes ? [{ id: "actgrads", label: "Activation gradients", detail: "CPU grads_acts 完整布局", bytes: activationGradientBytes }] : []),
        { id: "tokens", label: "Inputs + targets", detail: "两个 int32 token buffer", bytes: inputBytes },
      ];
      const total = buffers.reduce((sum, buffer) => sum + buffer.bytes, 0);

      const { b: B, t: T, c: C, l: L } = config;
      const transformerForward = 24 * B * T * L * C * C + 4 * B * L * T * T * C;
      const vocabProjection = 2 * B * T * C * VP;
      const work = {
        forward: transformerForward + vocabProjection,
        loss: 5 * B * T * VP,
        backward: 2.05 * (transformerForward + vocabProjection),
        optimizer: 20 * params,
      };
      const workTotal = Object.values(work).reduce((sum, value) => sum + value, 0);
      const shares = Object.fromEntries(Object.entries(work).map(([key, value]) => [key, (value / workTotal) * 100]));
      return { config, params, parameterBytes, gradientBytes, optimizerBytes, masterBytes, acts, inputBytes, activationGradientBytes, buffers, total, shares };
    }

    const phaseCatalog = {
      forward: {
        label: "Forward",
        purpose: "读取 inputs 与 parameters，写入 logits 和 backward 所需 activations。",
        cpu: ["encoder_forward：token + position embedding", "layernorm_forward：逐 token 统计与归一化", "matmul_forward：QKV、projection、MLP、词表投影", "attention_forward：causal scores、softmax、value aggregation", "gelu_forward / residual_forward：逐元素 C 函数"],
        cuda: ["encoder_forward_kernel3：embedding kernel", "layernorm / fused residual kernels", "cuBLASLt：QKV、projection、MLP 与词表 GEMM", "attention kernels；启用 cuDNN 时改为 cuDNN graph", "GELU kernel 或 cuBLASLt fused epilogue"],
      },
      loss: {
        label: "Loss",
        purpose: "把 B×T×Vp logits 与 targets 变为平均 cross-entropy，并产生 dlogits。",
        cpu: ["softmax_forward：生成 probs", "crossentropy_forward：读取 target 对应概率", "backward 起点设置为 1/(B·T)"],
        cuda: ["fused_classifier_kernel5：softmax + cross-entropy", "训练模式把 output buffer 原地改写为 dlogits", "FP32 losses 累加 micro-steps"],
      },
      backward: {
        label: "Backward",
        purpose: "逆序读取 activations 与 parameters，累加 parameter gradients。",
        cpu: ["crossentropy_softmax_backward", "matmul_backward / layernorm_backward", "attention_backward / gelu_backward / residual_backward", "encoder_backward：累加 wte 与 wpe 梯度"],
        cuda: ["cuBLASLt：矩阵乘输入、权重与 bias 梯度", "attention backward kernels 或 cuDNN graph", "layernorm / GELU / encoder backward kernels", "多 GPU 最后一个 micro-step 可触发 NCCL gradient reduction"],
      },
      optimizer: {
        label: "Optimizer",
        purpose: "AdamW 读取梯度与 m/v，写回参数和 optimizer state。",
        cpu: ["gpt2_update 遍历连续参数数组", "每个元素更新 FP32 m、v 与 param", "下一 step 读取改写后的 params_memory"],
        cuda: ["adamw kernel 并行更新参数分片", "m、v 始终使用 FP32", "BF16 可从 FP32 master weights 随机舍入写回 floatX", "ZeRO-1 时 optimizer state 按 rank 分片；实验不计分片"],
      },
    };

    function renderMetrics(result) {
      const modelBytes = result.parameterBytes + result.gradientBytes;
      $('[data-lc-metric="params"]').textContent = formatCount(result.params);
      $('[data-lc-metric="model"]').textContent = formatBytes(modelBytes);
      $('[data-lc-metric="acts"]').textContent = formatBytes(result.acts.total + result.activationGradientBytes);
      $('[data-lc-metric="total"]').textContent = formatBytes(result.total);
      $('[data-lc-model-note]').textContent = `${result.config.effectiveDtype.toUpperCase()} parameters + gradients`;
      $('[data-lc-acts-note]').textContent = result.config.path === "cpu" ? "forward acts + 同尺寸 grads_acts" : "non-cuDNN、recompute=0";
    }

    function renderMemory(result) {
      $('[data-lc-memory-total]').textContent = `Σ ${formatBytes(result.total)}`;
      const bar = $('[data-lc-memory-bar]');
      const list = $('[data-lc-memory-list]');
      bar.innerHTML = result.buffers.map((buffer) => {
        const share = (buffer.bytes / result.total) * 100;
        return `<span data-buffer="${buffer.id}" style="--size:${Math.max(share, 0.35)}" title="${buffer.label}: ${formatBytes(buffer.bytes)}"><i>${share >= 7 ? buffer.label : ""}</i></span>`;
      }).join("");
      bar.setAttribute("aria-label", result.buffers.map((buffer) => `${buffer.label} ${formatBytes(buffer.bytes)}`).join("；"));
      list.innerHTML = result.buffers.map((buffer) => {
        const share = (buffer.bytes / result.total) * 100;
        return `<div><i data-buffer="${buffer.id}"></i><span><b>${buffer.label}</b><small>${buffer.detail}</small></span><strong>${formatBytes(buffer.bytes)}</strong><em>${formatPercent(share)}</em></div>`;
      }).join("");
    }

    function renderShapes(result) {
      const { b: B, t: T, c: C, l: L, nh: NH } = result.config;
      const rows = [
        ["inputs / targets", `(${B}, ${T})`, "int32"],
        ["residual stream", `(${B}, ${T}, ${C})`, result.config.effectiveDtype.toUpperCase()],
        ["QKV per layer", `(${B}, ${T}, ${3 * C})`, result.config.effectiveDtype.toUpperCase()],
        ["attention scores", `(${B}, ${NH}, ${T}, ${T})`, result.config.effectiveDtype.toUpperCase()],
        ["MLP hidden", `(${B}, ${T}, ${4 * C})`, result.config.effectiveDtype.toUpperCase()],
        ["logits / dlogits", `(${B}, ${T}, ${VP})`, result.config.effectiveDtype.toUpperCase()],
        ["LayerNorm stats", `(${L}, ${B}, ${T})`, "FP32"],
      ];
      $('[data-lc-token-count]').textContent = `${(B * T).toLocaleString("zh-CN")} tokens / step`;
      $('[data-lc-shapes]').innerHTML = `<div class="lc-shape-table__head"><span>tensor</span><span>shape</span><span>dtype</span></div>${rows.map(([name, shape, dtype]) => `<div><b>${name}</b><code>${shape}</code><span>${dtype}</span></div>`).join("")}`;
    }

    function renderTimeline(result) {
      const phases = ["forward", "loss", "backward", "optimizer"];
      $('[data-lc-path-label]').textContent = result.config.path === "cuda" ? `CUDA · ${result.config.effectiveDtype.toUpperCase()}` : "CPU reference · FP32";
      $('[data-lc-timeline]').innerHTML = phases.map((id, index) => {
        const phase = phaseCatalog[id];
        const share = result.shares[id];
        return `<button type="button" data-lc-phase="${id}" aria-pressed="${activePhase === id}" style="--share:${Math.max(share, 9)}"><span>${index + 1}</span><b>${phase.label}</b><em>${formatPercent(share)}</em></button>`;
      }).join("");
      renderPhaseDetail(result);
    }

    function renderPhaseDetail(result) {
      const detail = $('[data-lc-phase-detail]');
      const phase = phaseCatalog[activePhase];
      const operations = phase[result.config.path];
      detail.innerHTML = `<div><span>当前阶段</span><h4>${phase.label}</h4><p>${phase.purpose}</p></div><ol>${operations.map((item) => `<li>${item}</li>`).join("")}</ol>`;
      $$('[data-lc-phase]').forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.lcPhase === activePhase)));
    }

    function renderAssumption(result) {
      const config = result.config;
      const dtypeNote = config.path === "cpu" && config.dtype === "bf16" ? "CPU reference 固定使用 FP32，因此 BF16 选择不改变当前内存。" : "";
      $('[data-lc-assumption]').textContent = `教学估算：Vp=${VP}，maxT=${config.maxT}，NH=${config.nh}（按 C/64），单卡；${config.path === "cuda" ? "CUDA activation 按 non-cuDNN、recompute=0，BF16 计 FP32 m/v 与 master weights。" : "CPU 计 forward activations 与同尺寸 activation gradients。"} ${dtypeNote} 不含 allocator、cuBLAS/cuDNN workspace、dataloader、checkpoint 和通信临时区。`;
    }

    function announce(result) {
      const path = result.config.path === "cuda" ? `CUDA ${result.config.effectiveDtype.toUpperCase()}` : "CPU reference FP32";
      $('[data-lc-live]').textContent = `${path} 教学估算已更新：${formatCount(result.params)} 参数，训练内存 ${formatBytes(result.total)}，activation 工作区 ${formatBytes(result.acts.total + result.activationGradientBytes)}。`;
    }

    function update({ speak = true } = {}) {
      lastResult = estimate(readConfig());
      renderMetrics(lastResult);
      renderMemory(lastResult);
      renderShapes(lastResult);
      renderTimeline(lastResult);
      renderAssumption(lastResult);
      if (speak) announce(lastResult);
    }

    lab.addEventListener("input", (event) => {
      if (event.target.matches("input, select")) update();
    });
    lab.addEventListener("change", (event) => {
      if (event.target.matches("input, select")) update();
    });
    lab.addEventListener("click", (event) => {
      const button = event.target.closest("[data-lc-phase]");
      if (!button) return;
      activePhase = button.dataset.lcPhase;
      renderPhaseDetail(lastResult);
      $('[data-lc-live]').textContent = `${phaseCatalog[activePhase].label}：${phaseCatalog[activePhase].purpose}`;
    });
    lab.addEventListener("keydown", (event) => {
      const current = event.target.closest("[data-lc-phase]");
      if (!current || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      const buttons = $$('[data-lc-phase]');
      let index = buttons.indexOf(current);
      if (event.key === "ArrowLeft") index = (index - 1 + buttons.length) % buttons.length;
      if (event.key === "ArrowRight") index = (index + 1) % buttons.length;
      if (event.key === "Home") index = 0;
      if (event.key === "End") index = buttons.length - 1;
      event.preventDefault();
      buttons[index].focus();
      buttons[index].click();
    });

    update({ speak: false });
  }
})();
