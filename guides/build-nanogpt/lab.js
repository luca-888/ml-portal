(() => {
  const roots = document.querySelectorAll('[data-guide="build-nanogpt"]');
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const format = new Intl.NumberFormat("zh-CN");

  const milestones = [
    {
      commit: "28916d9 · initial commit",
      title: "先写出 GPT-2 模型骨架",
      why: "先固定 attention、MLP、残差 block 和 GPTConfig，后续的数据与训练循环才有明确的输入输出接口。首个可见提交已经包含这些模块；README 所说的空文件起点属于课程叙事。",
      added: ["CausalSelfAttention", "MLP", "Block", "GPTConfig", "from_pretrained 骨架"],
      flow: [
        ["config", "12L · 12H · 768C"],
        ["token ids", "接口尚未接通"],
        ["Transformer", "模块已定义"],
        ["weights", "结构映射"],
      ],
    },
    {
      commit: "92b5bf9 → 7822fce",
      title: "用小文本定义 token batch 与 loss",
      why: "先用 Tiny Shakespeare 手工切出 B×T+1 个 token，证明 x/y 错位和交叉熵可以训练，再考虑可复用的数据加载器。",
      added: ["tiktoken", "Tiny Shakespeare", "x/y shift", "cross entropy", "50-step loop"],
      flow: [
        ["input.txt", "raw text"],
        ["tokenizer", "GPT-2 BPE"],
        ["x / y", "[B,T] · shift 1"],
        ["GPT", "logits"],
        ["loss", "cross entropy"],
      ],
    },
    {
      commit: "631f7d6 · add a DataLoaderLite",
      title: "把已验证的切片封装为状态",
      why: "训练循环需要连续取 batch。DataLoaderLite 保存 current_position，让 batch 生成逻辑与 optimizer loop 分离，同时保持 x/y 接口不变。",
      added: ["DataLoaderLite", "current_position", "next_batch", "boundary reset"],
      flow: [
        ["token stream", "1-D ids"],
        ["cursor", "+ B×T"],
        ["buffer", "B×T+1"],
        ["x / y", "[B,T]"],
        ["train loop", "next_batch()"],
      ],
    },
    {
      commit: "9ac321e · gpt-2 initialization",
      title: "用官方 GPT-2 checkpoint 检查结构",
      why: "自写 forward 已能训练和生成后，才有条件逐键核对 Hugging Face GPT-2 权重。四类 Conv1D 矩阵转置后复制，其余参数要求 shape 相同。",
      added: ["weight tying", "GPT-2 init", "state_dict 对齐", "Conv1D transpose", "shape assert"],
      flow: [
        ["HF state_dict", "Conv1D layout"],
        ["key filter", "忽略 mask buffer"],
        ["shape assert", "逐参数"],
        ["transpose", "4 类矩阵"],
        ["local GPT", "copy_"],
      ],
    },
    {
      commit: "5265b20 → 7230096",
      title: "逐项增加吞吐优化",
      why: "训练语义已经稳定，tokens/sec 才能成为反馈。TF32、bfloat16、compile、Flash Attention 和 vocab 对齐分别改变算术、dtype、执行图、kernel 与 shape。",
      added: ["TF32", "bfloat16 autocast", "torch.compile", "Flash Attention", "vocab 50304"],
      flow: [
        ["[B,T]", "token ids"],
        ["autocast", "bfloat16"],
        ["compiled GPT", "graph"],
        ["SDPA", "causal kernel"],
        ["tok/sec", "synchronized"],
      ],
    },
    {
      commit: "105f117 → 3a148e4",
      title: "把短循环扩成可训练的 optimizer step",
      why: "吞吐路径确定后，再加入参数分组、fused AdamW、梯度裁剪与 warmup/cosine 学习率，让长程训练的更新规则明确。",
      added: ["AdamW groups", "weight decay", "fused optimizer", "grad clipping", "LR schedule"],
      flow: [
        ["loss", "backward"],
        ["grad norm", "clip ≤ 1"],
        ["LR", "warmup · cosine"],
        ["AdamW", "fused if available"],
        ["parameters", "one update"],
      ],
    },
    {
      commit: "01be6b3 · add gradient accumulation",
      title: "在时间轴上拼出目标 token batch",
      why: "单次 forward 受显存限制。多个 micro-step 的 loss 先除以累积步数再 backward，最后只执行一次 optimizer.step，得到更大的有效 batch。",
      added: ["total_batch_size", "grad_accum_steps", "scaled loss", "micro-step loop"],
      flow: [
        ["micro 1", "loss / G"],
        ["micro 2…", "grad +="],
        ["micro G", "grad +="],
        ["clip", "once"],
        ["optimizer", "once"],
      ],
    },
    {
      commit: "ba2554a · DDP training",
      title: "在 GPU ranks 之间拼出同一个 global batch",
      why: "每个 rank 处理不同 token 窗口。前 G−1 个 micro-step 关闭梯度同步，最后一次 forward/backward 才 all-reduce；所有 rank 随后更新相同参数。",
      added: ["torchrun env", "NCCL process group", "ranked loader", "DDP wrapper", "last-step sync"],
      flow: [
        ["rank 0…N", "different windows"],
        ["local accum", "G micro-steps"],
        ["all-reduce", "last backward"],
        ["optimizer", "every rank"],
        ["rank 0", "logs"],
      ],
    },
    {
      commit: "69cb21f · switch to FineWeb EDU",
      title: "小文本换成可分片的训练数据",
      why: "模型、loader 和多卡步进已经有稳定接口，此时替换数据源只需让 loader 读取 train shard，并在边界切换到下一个文件。",
      added: ["fineweb.py", "edu_fineweb10B", "NumPy shards", "rank offsets", "shard rollover"],
      flow: [
        ["FineWeb EDU", "documents"],
        ["GPT-2 BPE", "uint16 ids"],
        ["train shards", "100M tokens each"],
        ["rank cursor", "strided windows"],
        ["x / y", "same contract"],
      ],
    },
    {
      commit: "21d3d32 → efedfac",
      title: "训练循环旁边增加测量与保存",
      why: "训练 loss 能下降以后，才加入独立 val split、HellaSwag、固定 prompt 生成、日志和 checkpoint，用不同信号判断训练状态。",
      added: ["val loader", "HellaSwag", "top-k samples", "log.txt", "checkpoint"],
      flow: [
        ["raw_model", "step N"],
        ["val", "mean loss"],
        ["HellaSwag", "accuracy"],
        ["generate", "sample text"],
        ["checkpoint", "rank 0"],
      ],
    },
  ];

  const numberFrom = (input, fallback) => {
    const value = Number(input?.value);
    return Number.isFinite(value) ? value : fallback;
  };

  roots.forEach((root) => {
    const lab = root.querySelector("[data-bng-lab]");
    if (!lab) return;

    const timeline = lab.querySelector("[data-bng-timeline]");
    const timelineButtons = Array.from(timeline.querySelectorAll("[data-bng-milestone]"));
    const milestoneCard = lab.querySelector("[data-bng-milestone-card]");
    const flow = lab.querySelector("[data-bng-flow]");
    const live = lab.querySelector("[data-bng-live]");
    const equation = lab.querySelector("[data-bng-equation]");
    const shapes = lab.querySelector("[data-bng-shapes]");
    const stepMap = lab.querySelector("[data-bng-step-map]");
    const inputs = Object.fromEntries(
      Array.from(lab.querySelectorAll("[data-bng-input]"), (input) => [input.dataset.bngInput, input]),
    );
    const values = Object.fromEntries(
      Array.from(lab.querySelectorAll("[data-bng-value]"), (output) => [output.dataset.bngValue, output]),
    );
    let activeMilestone = 0;

    const renderMilestone = (announce = true) => {
      const item = milestones[activeMilestone];
      timelineButtons.forEach((button, index) => {
        const active = index === activeMilestone;
        button.setAttribute("aria-pressed", String(active));
        button.tabIndex = active ? 0 : -1;
        button.dataset.state = active ? "active" : index < activeMilestone ? "past" : "next";
      });
      milestoneCard.innerHTML = `
        <div><span>里程碑 ${String(activeMilestone + 1).padStart(2, "0")} / ${milestones.length}</span><code>${item.commit}</code></div>
        <h4>${item.title}</h4>
        <p>${item.why}</p>
        <ul>${item.added.map((label) => `<li>${label}</li>`).join("")}</ul>
      `;
      flow.innerHTML = item.flow.map(([label, detail], index) => `
        <div style="--bng-flow-index:${index}"><span>${String(index + 1).padStart(2, "0")}</span><b>${label}</b><small>${detail}</small></div>
        ${index < item.flow.length - 1 ? "<i aria-hidden=\"true\">→</i>" : ""}
      `).join("");
      if (announce) live.textContent = `已切换到里程碑 ${activeMilestone + 1}：${item.title}。${item.why}`;
    };

    const readBatch = () => ({
      batch: Math.round(numberFrom(inputs.batch, 8)),
      sequence: Math.round(numberFrom(inputs.sequence, 1024)),
      world: Math.round(numberFrom(inputs.world, 4)),
      accumulation: Math.round(numberFrom(inputs.accumulation, 16)),
    });

    const renderBatch = (announce = true) => {
      const plan = readBatch();
      Object.entries(plan).forEach(([key, value]) => {
        values[key].textContent = format.format(value);
      });

      const localTokens = plan.batch * plan.sequence;
      const parallelTokens = localTokens * plan.world;
      const globalTokens = parallelTokens * plan.accumulation;
      const bufferTokens = localTokens + 1;
      equation.innerHTML = `
        <div><span>每 rank / micro-step</span><b>${format.format(plan.batch)} × ${format.format(plan.sequence)}</b><strong>${format.format(localTokens)} tokens</strong></div>
        <i>×</i>
        <div><span>并行 ranks</span><b>world ${plan.world}</b><strong>${format.format(parallelTokens)} tokens</strong></div>
        <i>×</i>
        <div><span>时间累积</span><b>${plan.accumulation} micro-steps</b><strong>${format.format(globalTokens)} tokens</strong></div>
      `;

      shapes.innerHTML = `
        <div class="bng-buffer-shape"><span>loader slice</span><b>${format.format(bufferTokens)}</b><small>B×T+1 token / rank / micro-step</small><i style="--bng-fill:${Math.min(100, 18 + Math.log2(bufferTokens) * 5)}%"></i></div>
        <div class="bng-shape-card is-input"><span>input x</span><b>[${plan.batch}, ${plan.sequence}]</b><small>buffer[:-1]</small><div><i>t₀</i><i>t₁</i><i>…</i><i>tₜ₋₁</i></div></div>
        <div class="bng-shift-arrow" aria-hidden="true"><b>+1</b><i>→</i></div>
        <div class="bng-shape-card is-target"><span>target y</span><b>[${plan.batch}, ${plan.sequence}]</b><small>buffer[1:]</small><div><i>t₁</i><i>t₂</i><i>…</i><i>tₜ</i></div></div>
      `;

      const cells = Array.from({ length: plan.accumulation }, (_, index) => {
        const final = index === plan.accumulation - 1;
        return `<i class="${final ? "is-sync" : ""}" title="micro-step ${index + 1}${final ? "：DDP 同步" : "：本地累积"}"><span>${index + 1}</span></i>`;
      }).join("");
      const ranks = Array.from({ length: plan.world }, (_, rank) => `
        <div class="bng-rank-row">
          <b>rank ${rank}</b>
          <div style="--bng-steps:${plan.accumulation}">${cells}</div>
          <em>${rank === 0 ? "sync · log" : "sync"}</em>
        </div>
      `).join("");
      stepMap.innerHTML = `
        <header><div><span>一次 global step</span><b>${plan.accumulation}× forward + backward / rank</b></div><strong>最后一次 backward → all-reduce → clip → AdamW</strong></header>
        <div class="bng-rank-grid">${ranks}</div>
        <footer><span><i></i>本地累积，不通信</span><span><i class="is-sync"></i>最后一个 micro-step，同步梯度</span></footer>
      `;
      if (announce) {
        live.textContent = `batch 实验已更新：输入和目标 shape 都是 ${plan.batch} 乘 ${plan.sequence}，world size ${plan.world}，累积 ${plan.accumulation} 步，global token batch 为 ${format.format(globalTokens)}；DDP 在第 ${plan.accumulation} 个 micro-step 同步。`;
      }
    };

    timeline.addEventListener("click", (event) => {
      const button = event.target.closest("[data-bng-milestone]");
      if (!button) return;
      activeMilestone = Number(button.dataset.bngMilestone);
      renderMilestone();
    });

    timeline.addEventListener("keydown", (event) => {
      const button = event.target.closest("[data-bng-milestone]");
      if (!button || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const current = Number(button.dataset.bngMilestone);
      if (event.key === "Home") activeMilestone = 0;
      if (event.key === "End") activeMilestone = milestones.length - 1;
      if (event.key === "ArrowLeft") activeMilestone = (current - 1 + milestones.length) % milestones.length;
      if (event.key === "ArrowRight") activeMilestone = (current + 1) % milestones.length;
      timelineButtons[activeMilestone].focus();
      renderMilestone();
    });

    lab.addEventListener("input", (event) => {
      if (!event.target.closest("[data-bng-input]")) return;
      renderBatch();
    });

    const updateMotionPreference = () => {
      lab.toggleAttribute("data-reduced-motion", reducedMotion.matches);
    };
    if (typeof reducedMotion.addEventListener === "function") {
      reducedMotion.addEventListener("change", updateMotionPreference);
    }

    updateMotionPreference();
    renderMilestone(false);
    renderBatch(false);
  });
})();
