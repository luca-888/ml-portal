(() => {
  document.querySelectorAll('[data-guide="liger-kernel"]').forEach(initLigerLab);

  function initLigerLab(root) {
    const lab = root.querySelector("[data-lk-lab]");
    if (!lab) return;

    const get = (selector) => lab.querySelector(selector);
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const controls = {
      kernel: get("[data-lk-kernel]"),
      rows: get("[data-lk-rows]"),
      rowsOutput: get("[data-lk-rows-output]"),
      hidden: get("[data-lk-hidden]"),
      vocab: get("[data-lk-vocab]"),
      vocabControl: get("[data-lk-vocab-control]"),
      block: get("[data-lk-block]"),
      previous: get("[data-lk-prev]"),
      next: get("[data-lk-next]"),
      play: get("[data-lk-play]"),
    };

    const kernels = {
      rmsnorm: {
        title: "RMSNorm · row reduction",
        width: (state) => state.hidden,
        stages: [
          { short: "load", copy: "一个 program 读取一行 x 与权重 w；尾部 offsets 由 mask 关闭。", alu: ["tl.load", "cast fp32"] },
          { short: "reduce", copy: "tile 内计算 x² 并横向归约，得到这一行的 mean square。", alu: ["x × x", "tl.sum", "÷ H"] },
          { short: "normalize", copy: "计算 rstd，并把这个标量广播回整行完成缩放。", alu: ["rsqrt", "x × rstd", "× weight"] },
          { short: "store", copy: "只写最终 y 和每行一个 rstd；不写逐元素归一化中间张量。", alu: ["store y", "store rstd"] },
        ],
        eager: [
          ["x", 0], ["square(x)", 1], ["mean", 1], ["rsqrt", 2], ["x · rstd", 2], ["· weight", 2], ["y", 3],
        ],
        fused: [
          ["load x + w", 0], ["square + reduce", 1], ["rsqrt + scale", 2], ["store y + rstd", 3],
        ],
        hbm: [
          ["x", "w"], ["x tile"], ["rstd"], ["y", "rstd"],
        ],
      },
      swiglu: {
        title: "SwiGLU · elementwise fusion",
        width: (state) => state.hidden,
        stages: [
          { short: "load", copy: "相同 offsets 同时加载 gate a 与 up projection b。", alu: ["load a", "load b"] },
          { short: "activate", copy: "a 转为 fp32 后计算 sigmoid 与 SiLU，结果保持在片上。", alu: ["sigmoid(a)", "a × sigmoid"] },
          { short: "gate", copy: "SiLU(a) 与 b 相乘；不物化独立的 SiLU 激活张量。", alu: ["silu(a) × b"] },
          { short: "store", copy: "最终 c 写回 HBM；backward 会从 a 重算 sigmoid。", alu: ["store c", "recompute in bwd"] },
        ],
        eager: [
          ["a", 0], ["sigmoid(a)", 1], ["a · sigmoid", 1], ["silu temp", 2], ["× b", 2], ["c", 3],
        ],
        fused: [
          ["load a + b", 0], ["sigmoid + silu", 1], ["gate multiply", 2], ["store c", 3],
        ],
        hbm: [
          ["a", "b"], ["a + b tile"], ["c tile"], ["c"],
        ],
      },
      flce: {
        title: "Fused Linear CE · token chunking",
        width: (state) => state.vocab,
        stages: [
          { short: "chunk", copy: "按 V/H 选择 token chunk；任一时刻只处理一条 BT 子区间。", alu: ["slice BT", "choose chunk"] },
          { short: "linear", copy: "当前 hidden chunk 与词表权重相乘，临时得到 chunk × V logits。", alu: ["Xchunk @ Wᵀ", "logits chunk"] },
          { short: "loss → grad", copy: "cross entropy 在 logits buffer 上写入梯度，使同一块内存立即进入反向。", alu: ["logsumexp", "loss", "in-place dlogits"] },
          { short: "backprop", copy: "dlogits 投影为 grad_input，并累加 grad_weight；随后复用缓冲区处理下一块。", alu: ["dX", "dW +=", "next chunk"] },
        ],
        eager: [
          ["BT × H", 0], ["linear", 1], ["BT × V logits", 1], ["cross entropy", 2], ["dlogits", 2], ["backward", 3],
        ],
        fused: [
          ["token chunk", 0], ["chunk × V", 1], ["loss → dlogits", 2], ["dX + dW → next", 3],
        ],
        hbm: [
          ["hidden", "weight"], ["logits chunk"], ["loss", "dlogits chunk"], ["dX", "dW"],
        ],
      },
    };

    let step = 0;
    let timer = null;

    function nextPowerOfTwo(value) {
      return 2 ** Math.ceil(Math.log2(Math.max(1, value)));
    }

    function state() {
      return {
        kernel: controls.kernel.value,
        rows: Number(controls.rows.value),
        hidden: Number(controls.hidden.value),
        vocab: Number(controls.vocab.value),
        block: Number(controls.block.value),
        bytes: 2,
      };
    }

    function formatCount(value) {
      return new Intl.NumberFormat("zh-CN").format(value);
    }

    function formatBytes(value) {
      if (value <= 0) return "0 B";
      const units = ["B", "KiB", "MiB", "GiB", "TiB"];
      const index = Math.min(units.length - 1, Math.floor(Math.log(value) / Math.log(1024)));
      const scaled = value / 1024 ** index;
      return `${scaled >= 100 ? scaled.toFixed(0) : scaled >= 10 ? scaled.toFixed(1) : scaled.toFixed(2)} ${units[index]}`;
    }

    function chunkInfo(current) {
      const increase = Math.ceil(current.vocab / current.hidden);
      const chunkSize = nextPowerOfTwo(Math.ceil(current.rows / increase));
      return {
        increase,
        chunkSize,
        chunks: Math.ceil(current.rows / chunkSize),
      };
    }

    function renderStages(config) {
      get("[data-lk-stages]").innerHTML = config.stages.map((stage, index) =>
        `<button type="button" role="tab" aria-selected="${index === step}" data-lk-stage="${index}"${index === step ? " data-active" : ""}><span>${String(index + 1).padStart(2, "0")}</span>${stage.short}</button>`
      ).join("");
    }

    function renderGraph(config) {
      const makeLane = (label, nodes, fused) => `<div class="lk-graph-lane" data-kind="${fused ? "fused" : "eager"}">
        <header><b>${label}</b><span>${fused ? "one managed lifetime" : "operator boundaries"}</span></header>
        <div>${nodes.map(([name, stage], index) => `<span data-stage="${stage}"${stage === step ? " data-active" : ""}${stage < step ? " data-done" : ""}><i>${index + 1}</i>${name}</span>`).join("<em>→</em>")}</div>
      </div>`;
      get("[data-lk-graphs]").innerHTML = makeLane("PyTorch graph", config.eager, false) + makeLane("Liger path", config.fused, true);
    }

    function renderDevice(config, current) {
      const workWidth = config.width(current);
      const tiles = Math.ceil(workWidth / current.block);
      const tileIndex = Math.max(0, tiles - 1);
      const active = workWidth - tileIndex * current.block || current.block;
      const shown = 32;
      const shownActive = Math.max(1, Math.min(shown, Math.ceil(active / current.block * shown)));
      const laneStart = tileIndex * current.block;
      const lanes = Array.from({ length: shown }, (_, index) => {
        const isActive = index < shownActive;
        const logicalOffset = laneStart + Math.floor(index / shown * current.block);
        return `<span data-state="${isActive ? "active" : "masked"}" title="${isActive ? `offset ${logicalOffset}` : "masked tail"}"><i>${index}</i><b>${isActive ? logicalOffset : "×"}</b></span>`;
      }).join("");
      const utilization = active / current.block * 100;
      const stage = config.stages[step];

      get("[data-lk-program]").textContent = `tile ${tileIndex}/${tiles - 1} · offsets ${formatCount(laneStart)}…`;
      get("[data-lk-util]").textContent = `${utilization.toFixed(utilization < 10 ? 1 : 0)}% active`;
      get("[data-lk-lanes]").innerHTML = lanes;
      get("[data-lk-alu]").innerHTML = stage.alu.map((operation, index) => `<span style="--op:${index}"${index === stage.alu.length - 1 ? " data-hot" : ""}>${operation}</span>`).join("");
      get("[data-lk-hbm]").dataset.step = String(step);
      get("[data-lk-hbm]").querySelector("div").innerHTML = config.hbm[step].map((name) => `<span>${name}</span>`).join("");
      get("[data-lk-microscope]").dataset.step = String(step);
    }

    function bar(label, value, max, detail, kind) {
      const raw = max > 0 ? value / max : 0;
      const width = value === 0 ? 0 : Math.max(2, raw * 100);
      return `<div class="lk-memory-row" data-kind="${kind}"><header><b>${label}</b><span>${detail}</span><strong>${formatBytes(value)}</strong></header><div><i style="width:${width}%"></i></div></div>`;
    }

    function metric(label, value, note) {
      return `<div><dt>${label}</dt><dd>${value}</dd><small>${note}</small></div>`;
    }

    function renderMemory(current) {
      const tensor = current.rows * current.hidden * current.bytes;
      let title;
      let baseline;
      let liger;
      let baselineDetail;
      let ligerDetail;
      let metrics;

      if (current.kernel === "flce") {
        const info = chunkInfo(current);
        baseline = current.rows * current.vocab * current.bytes;
        liger = info.chunkSize * current.vocab * current.bytes;
        title = "logits buffer footprint";
        baselineDetail = `${formatCount(current.rows)} × ${formatCount(current.vocab)} × 2 B`;
        ligerDetail = `${formatCount(info.chunkSize)} × ${formatCount(current.vocab)} × 2 B`;
        metrics = [
          ["V / H", `${info.increase}×`, "ceil(V/H)"],
          ["chunk size", formatCount(info.chunkSize), "next power of 2"],
          ["chunks", String(info.chunks), "ceil(BT/chunk)"],
          ["logits ratio", `${(baseline / liger).toFixed(1)}×`, "完整 / 分块"],
        ];
      } else if (current.kernel === "rmsnorm") {
        baseline = tensor * 2;
        liger = current.rows * 4;
        title = "额外中间状态估算";
        baselineDetail = "两个同形逐元素临时张量";
        ligerDetail = "每行一个 fp32 rstd";
        metrics = [
          ["rows", formatCount(current.rows), "独立归一化行"],
          ["H", formatCount(current.hidden), "每行元素"],
          ["tile slices", String(Math.ceil(current.hidden / current.block)), "页面检查宽度"],
          ["stored scalars", formatCount(current.rows), "backward rstd"],
        ];
      } else {
        baseline = tensor;
        liger = 0;
        title = "forward 额外激活估算";
        baselineDetail = "物化 SiLU(a)";
        ligerDetail = "片上消费，不单独写回";
        metrics = [
          ["elements", formatCount(current.rows * current.hidden), "逐元素 gate"],
          ["eager temp", "1 tensor", "SiLU(a)"],
          ["fused temp", "0 tensor", "仅计额外激活"],
          ["backward", "recompute", "sigmoid + silu"],
        ];
      }

      get("[data-lk-memory-title]").textContent = title;
      get("[data-lk-memory-bars]").innerHTML = bar("operator graph", baseline, baseline, baselineDetail, "eager") + bar("Liger path", liger, baseline, ligerDetail, "fused");
      get("[data-lk-metrics]").innerHTML = metrics.map((item) => metric(...item)).join("");
    }

    function status(config, current) {
      const stage = config.stages[step];
      const suffix = current.kernel === "flce"
        ? ` 当前估算：${chunkInfo(current).chunks} 个 token chunks。`
        : ` 当前检查最后一个宽度为 ${current.block} 的 tile。`;
      return `${step + 1}/${config.stages.length} · ${stage.short} — ${stage.copy}${suffix}`;
    }

    function render(announce = false) {
      const current = state();
      const config = kernels[current.kernel];
      step = Math.min(step, config.stages.length - 1);
      controls.rowsOutput.textContent = formatCount(current.rows);
      controls.vocabControl.hidden = current.kernel !== "flce";
      get("[data-lk-title]").textContent = config.title;
      get("[data-lk-stage-copy]").textContent = config.stages[step].copy;
      renderStages(config);
      renderGraph(config);
      renderDevice(config, current);
      renderMemory(current);
      const message = status(config, current);
      get("[data-lk-status]").textContent = message;
      if (announce) get("[data-lk-live]").textContent = message;
    }

    function stop() {
      if (timer) window.clearInterval(timer);
      timer = null;
      controls.play.setAttribute("aria-pressed", "false");
      controls.play.textContent = "播放";
    }

    function move(delta, announce = true) {
      const total = kernels[state().kernel].stages.length;
      step = (step + delta + total) % total;
      render(announce);
    }

    function togglePlay() {
      if (timer) {
        stop();
        return;
      }
      controls.play.setAttribute("aria-pressed", "true");
      controls.play.textContent = "暂停";
      if (reducedMotion) {
        move(1);
        stop();
        return;
      }
      timer = window.setInterval(() => move(1, true), 1250);
    }

    lab.addEventListener("input", (event) => {
      if (!event.target.matches("input, select")) return;
      if (event.target === controls.kernel) {
        step = 0;
        stop();
      }
      render(true);
    });

    lab.addEventListener("click", (event) => {
      const stageButton = event.target.closest("[data-lk-stage]");
      if (stageButton) {
        step = Number(stageButton.dataset.lkStage);
        stop();
        render(true);
      }
    });

    get("[data-lk-stages]").addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      move(event.key === "ArrowRight" ? 1 : -1);
      get(`[data-lk-stage="${step}"]`)?.focus();
    });

    controls.previous.addEventListener("click", () => { stop(); move(-1); });
    controls.next.addEventListener("click", () => { stop(); move(1); });
    controls.play.addEventListener("click", togglePlay);

    render();
  }
})();
