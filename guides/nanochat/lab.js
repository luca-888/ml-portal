(() => {
  const guideRoots = document.querySelectorAll('[data-guide="nanochat"]');
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  const clamp = (value, min, max) => Math.min(Math.max(value, min), max);
  const numberFrom = (input, fallback) => {
    const value = Number(input?.value);
    return Number.isFinite(value) ? value : fallback;
  };
  const compact = new Intl.NumberFormat("zh-CN", {
    notation: "compact",
    maximumFractionDigits: 2,
  });
  const decimal = new Intl.NumberFormat("zh-CN", {
    maximumFractionDigits: 1,
  });

  const vocabSize = 32768;
  const tokenRatio = 8;
  const referenceDepth = 24;
  const referenceGpus = 8;
  const referencePretrainHours = 1.65;

  const modelAt = (depth) => {
    const width = Math.ceil((depth * 64) / 128) * 128;
    const heads = width / 128;
    const transformerMatrices = depth * 12 * width * width;
    const lmHead = vocabSize * width;
    const scalingParams = transformerMatrices + lmHead;
    const valueEmbeddingCount = Math.ceil(depth / 2);
    const totalParams = transformerMatrices
      + ((2 + valueEmbeddingCount) * vocabSize * width);
    return {
      depth,
      width,
      heads,
      scalingParams,
      totalParams,
      targetTokens: scalingParams * tokenRatio,
    };
  };

  const referenceModel = modelAt(referenceDepth);
  const stages = [
    { id: "tokenizer", label: "tokenizer / data", ratio: 0.08, color: "#f2a65a" },
    { id: "pretrain", label: "base pretraining", ratio: 0.72, color: "#e85d2a" },
    { id: "sft", label: "SFT", ratio: 0.09, color: "#7c3aed" },
    { id: "eval", label: "evaluation", ratio: 0.10, color: "#2563eb" },
    { id: "chat", label: "chat smoke test", ratio: 0.01, color: "#0f766e" },
  ];
  const flowDetails = [
    {
      title: "训练 tokenizer",
      command: "scripts/tok_train.py",
      input: "ClimbMix parquet 文本",
      output: "tokenizer/ · token_bytes.pt",
      note: "固定 32,768 BPE 词表；后续三个阶段共享同一套 token id。",
    },
    {
      title: "训练 base model",
      command: "scripts/base_train.py",
      input: "文本 shards + tokenizer",
      output: "base_checkpoints/<tag>/",
      note: "保存 model、meta、dataloader state，并按 rank 保存 optimizer state。",
    },
    {
      title: "监督微调",
      command: "scripts/chat_sft.py",
      input: "base checkpoint + SFT mixture",
      output: "chatsft_checkpoints/<tag>/",
      note: "继承 base 配置；conversation loss mask 决定需要学习的 token。",
    },
    {
      title: "分阶段评估",
      command: "base_eval.py · chat_eval.py",
      input: "base 或 SFT checkpoint",
      output: "BPB · CORE · ChatCORE · samples",
      note: "评估读取 checkpoint，不把分数写回模型参数。",
    },
    {
      title: "运行对话",
      command: "scripts/chat_cli.py",
      input: "SFT checkpoint + prompt",
      output: "conversation token state",
      note: "Engine 使用 KV cache 生成；clear 只重置当前会话历史。",
    },
  ];

  guideRoots.forEach((guide) => {
    const lab = guide.querySelector("[data-nanochat-lab]");
    if (!lab) return;

    const inputs = Object.fromEntries(
      Array.from(lab.querySelectorAll("[data-nanochat-input]"), (input) => [
        input.dataset.nanochatInput,
        input,
      ]),
    );
    const values = Object.fromEntries(
      Array.from(lab.querySelectorAll("[data-nanochat-value]"), (output) => [
        output.dataset.nanochatValue,
        output,
      ]),
    );
    const metrics = lab.querySelector("[data-nanochat-metrics]");
    const coverage = lab.querySelector("[data-nanochat-coverage]");
    const stageBar = lab.querySelector("[data-nanochat-stage-bar]");
    const stageList = lab.querySelector("[data-nanochat-stage-list]");
    const syncNote = lab.querySelector("[data-nanochat-sync-note]");
    const live = lab.querySelector("[data-nanochat-live]");
    const stepper = lab.querySelector("[data-nanochat-stepper]");
    const stepButtons = Array.from(stepper.querySelectorAll("[data-nanochat-step]"));
    const flowDetail = lab.querySelector("[data-nanochat-flow-detail]");

    let budgetAnchor = "hours";
    let activeStep = 0;

    const synchronizeBudget = () => {
      const gpus = Math.round(numberFrom(inputs.gpus, 8));
      const price = clamp(numberFrom(inputs.price, 3), 0.1, 30);
      inputs.price.value = String(price);
      const hourlyClusterCost = gpus * price;

      if (budgetAnchor === "budget") {
        const requestedBudget = numberFrom(inputs.budget, 48);
        const hours = clamp(requestedBudget / hourlyClusterCost, 0.5, 48);
        inputs.hours.value = String(Math.round(hours * 2) / 2);
        inputs.budget.value = String(Math.round(numberFrom(inputs.hours, 2) * hourlyClusterCost));
      } else {
        const hours = numberFrom(inputs.hours, 2);
        const budget = hours * hourlyClusterCost;
        inputs.budget.max = String(Math.max(1000, Math.ceil(budget / 100) * 100));
        inputs.budget.value = String(Math.round(budget));
      }
    };

    const readPlan = () => {
      synchronizeBudget();
      return {
        depth: Math.round(numberFrom(inputs.depth, 24)),
        gpus: Math.round(numberFrom(inputs.gpus, 8)),
        hours: numberFrom(inputs.hours, 2),
        budget: numberFrom(inputs.budget, 48),
        price: numberFrom(inputs.price, 3),
      };
    };

    const renderValues = (plan) => {
      values.depth.textContent = String(plan.depth);
      values.gpus.textContent = String(plan.gpus);
      values.hours.textContent = `${plan.hours.toFixed(1)} h`;
      values.budget.textContent = `$${Math.round(plan.budget)}`;
      syncNote.textContent = budgetAnchor === "budget"
        ? "当前按预算反算可用时长；拖动计划时长后会重新计算预算。"
        : "当前按计划时长计算预算；拖动预算后会反算可用时长。";
    };

    const renderModel = (plan, model) => {
      const relativeCompute = (model.scalingParams / referenceModel.scalingParams) ** 2;
      const estimatedPretrainHours = referencePretrainHours
        * relativeCompute
        * (referenceGpus / plan.gpus);
      const allocatedPretrainHours = plan.hours * stages[1].ratio;
      const coverageRatio = allocatedPretrainHours / estimatedPretrainHours;
      const coveragePercent = coverageRatio * 100;
      const checkpointGb = (model.totalParams * 4) / 1e9;

      metrics.innerHTML = `
        <div><span>model width</span><b>${model.width.toLocaleString("zh-CN")}</b><small>depth × 64，向上对齐 128</small></div>
        <div><span>attention heads</span><b>${model.heads}</b><small>head dim = 128</small></div>
        <div><span>近似总参数</span><b>${compact.format(model.totalParams)}</b><small>含 token / value embeddings</small></div>
        <div><span>训练 token horizon</span><b>${compact.format(model.targetTokens)}</b><small>教学模型使用 ratio = ${tokenRatio}</small></div>
        <div><span>FP32 模型权重</span><b>${decimal.format(checkpointGb)} GB</b><small>不含 optimizer 与运行时张量</small></div>
      `;

      let state = "不足";
      let stateClass = "is-low";
      if (coverageRatio >= 1) {
        state = "达到教学参考量";
        stateClass = "is-ready";
      } else if (coverageRatio >= 0.75) {
        state = "接近教学参考量";
        stateClass = "is-close";
      }
      const displayedCoverage = Math.min(coveragePercent, 100);
      coverage.className = `nanochat-coverage ${stateClass}`;
      coverage.innerHTML = `
        <div class="nanochat-coverage-head">
          <div><span>pretraining compute coverage</span><b>${state}</b></div>
          <strong>${decimal.format(coveragePercent)}%</strong>
        </div>
        <div class="nanochat-coverage-track" aria-label="教学预训练算力覆盖 ${decimal.format(coveragePercent)}%"><i style="--nanochat-coverage:${displayedCoverage}%"></i></div>
        <p>本计划为 pretraining 分配 ${allocatedPretrainHours.toFixed(2)} 小时；同等级 GPU 的简化参考需求为 ${decimal.format(estimatedPretrainHours)} 小时。估算未考虑显存不足、通信、kernel、dtype、评估噪声和数据速度。</p>
      `;
      return { coveragePercent, estimatedPretrainHours };
    };

    const renderStages = (plan) => {
      stageBar.innerHTML = stages.map((stage) => `
        <span style="--nanochat-stage-width:${stage.ratio * 100}%;--nanochat-stage-color:${stage.color}" title="${stage.label} ${stage.ratio * 100}%"><b>${stage.ratio >= 0.08 ? stage.label : ""}</b></span>
      `).join("");
      stageList.innerHTML = stages.map((stage, index) => {
        const hours = plan.hours * stage.ratio;
        return `<div><span style="--nanochat-stage-color:${stage.color}">${String(index + 1).padStart(2, "0")}</span><b>${stage.label}</b><strong>${hours < 0.1 ? `${Math.max(1, Math.round(hours * 60))} min` : `${hours.toFixed(2)} h`}</strong><small>${Math.round(stage.ratio * 100)}%</small></div>`;
      }).join("");
    };

    const renderFlow = (announce = true) => {
      stepButtons.forEach((button, index) => {
        const state = index < activeStep ? "complete" : index === activeStep ? "active" : "upcoming";
        button.dataset.state = state;
        button.setAttribute("aria-pressed", String(index === activeStep));
      });

      const detail = flowDetails[activeStep];
      flowDetail.innerHTML = `
        <div><span>当前阶段 ${String(activeStep + 1).padStart(2, "0")}</span><h4>${detail.title}</h4><code>${detail.command}</code></div>
        <dl>
          <div><dt>输入</dt><dd>${detail.input}</dd></div>
          <div><dt>产物</dt><dd>${detail.output}</dd></div>
        </dl>
        <p>${detail.note}</p>
      `;
      if (announce) live.textContent = `已选择${detail.title}。输入：${detail.input}。产物：${detail.output}。`;
    };

    const update = (announce = true) => {
      const plan = readPlan();
      const model = modelAt(plan.depth);
      renderValues(plan);
      const result = renderModel(plan, model);
      renderStages(plan);
      lab.toggleAttribute("data-reduced-motion", reducedMotion.matches);
      if (announce) {
        live.textContent = `教学估算已更新：depth ${plan.depth}，${plan.gpus} 张 GPU，近似 ${compact.format(model.totalParams)} 参数，计划 ${plan.hours.toFixed(1)} 小时，预算 ${Math.round(plan.budget)} 美元，预训练覆盖 ${decimal.format(result.coveragePercent)}%。`;
      }
    };

    lab.addEventListener("input", (event) => {
      const input = event.target.closest("[data-nanochat-input]");
      if (!input) return;
      const name = input.dataset.nanochatInput;
      if (name === "budget") budgetAnchor = "budget";
      if (name === "hours") budgetAnchor = "hours";
      update();
    });

    stepper.addEventListener("click", (event) => {
      const button = event.target.closest("[data-nanochat-step]");
      if (!button) return;
      activeStep = Number(button.dataset.nanochatStep);
      renderFlow();
    });

    stepper.addEventListener("keydown", (event) => {
      const button = event.target.closest("[data-nanochat-step]");
      if (!button || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const currentIndex = Number(button.dataset.nanochatStep);
      if (event.key === "Home") activeStep = 0;
      if (event.key === "End") activeStep = stepButtons.length - 1;
      if (event.key === "ArrowLeft") activeStep = (currentIndex - 1 + stepButtons.length) % stepButtons.length;
      if (event.key === "ArrowRight") activeStep = (currentIndex + 1) % stepButtons.length;
      stepButtons[activeStep].focus();
      renderFlow();
    });

    const onMotionChange = () => lab.toggleAttribute("data-reduced-motion", reducedMotion.matches);
    if (typeof reducedMotion.addEventListener === "function") reducedMotion.addEventListener("change", onMotionChange);

    update(false);
    renderFlow(false);
  });
})();
