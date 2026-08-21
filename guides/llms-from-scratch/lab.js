(() => {
  const STAGES = [
    { name: "Token IDs", key: "tokens" },
    { name: "Embeddings", key: "embed" },
    { name: "Q / K / V", key: "qkv" },
    { name: "Attention scores", key: "scores" },
    { name: "Mask + softmax", key: "prob" },
    { name: "Context + residual", key: "residual" },
    { name: "Logits + loss", key: "loss" },
  ];

  document.querySelectorAll('[data-guide="llms-from-scratch"]').forEach(init);

  function init(root) {
    const lab = root.querySelector("[data-lfs-lab]");
    if (!lab) return;

    const $ = (selector) => lab.querySelector(selector);
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let state = { stage: 0, matrix: "prob", model: null };

    lab.tabIndex = 0;

    function tokenize(value) {
      const tokens = value.trim().split(/\s+/u).filter(Boolean).slice(0, 6);
      return tokens.length ? tokens : ["token"];
    }

    function tokenId(token) {
      let hash = 2166136261;
      for (const char of token) {
        hash ^= char.codePointAt(0);
        hash = Math.imul(hash, 16777619);
      }
      return (hash >>> 0) % 997 + 1;
    }

    const round = (value) => Math.round(value * 1000) / 1000;
    const vector = (length, fn) => Array.from({ length }, (_, index) => fn(index));
    const dot = (left, right) => left.reduce((sum, value, index) => sum + value * right[index], 0);

    function project(input, phase) {
      const width = input.length;
      return vector(width, (out) => {
        const sum = input.reduce((total, value, index) => {
          const weight = Math.sin((index + 1) * (out + 1) * phase) / Math.sqrt(width);
          return total + value * weight;
        }, 0);
        return round(sum);
      });
    }

    function softmax(row) {
      const finite = row.filter(Number.isFinite);
      const max = Math.max(...finite);
      const exps = row.map((value) => Number.isFinite(value) ? Math.exp(value - max) : 0);
      const total = exps.reduce((sum, value) => sum + value, 0);
      return exps.map((value) => value / total);
    }

    function buildModel() {
      const tokens = tokenize($("[data-tokens]").value);
      $("[data-tokens]").value = tokens.join(" ");
      const ids = tokens.map(tokenId);
      const dim = Number($("[data-dim]").value);
      const heads = Number($("[data-heads]").value);
      const headDim = dim / heads;
      const causal = $("[data-causal]").checked;

      const embeddings = ids.map((id, position) => vector(dim, (axis) => round(
        0.68 * Math.sin((id + 1) * (axis + 1) * 0.017) +
        0.32 * Math.cos((position + 1) * (axis + 2) * 0.31)
      )));
      const q = embeddings.map((row) => project(row, 0.47));
      const k = embeddings.map((row) => project(row, 0.31));
      const v = embeddings.map((row) => project(row, 0.23));
      const contexts = tokens.map(() => Array(dim).fill(0));
      let firstScores = [];
      let firstMasked = [];
      let firstProb = [];

      for (let head = 0; head < heads; head += 1) {
        const start = head * headDim;
        const end = start + headDim;
        const scores = q.map((query) => k.map((key) => round(
          dot(query.slice(start, end), key.slice(start, end)) / Math.sqrt(headDim)
        )));
        const masked = scores.map((row, queryIndex) => row.map((score, keyIndex) =>
          causal && keyIndex > queryIndex ? -Infinity : score
        ));
        const probabilities = masked.map(softmax);

        probabilities.forEach((row, queryIndex) => {
          for (let axis = start; axis < end; axis += 1) {
            contexts[queryIndex][axis] = round(row.reduce(
              (sum, probability, keyIndex) => sum + probability * v[keyIndex][axis], 0
            ));
          }
        });

        if (head === 0) {
          firstScores = scores;
          firstMasked = masked;
          firstProb = probabilities;
        }
      }

      const residual = embeddings.map((row, position) => row.map(
        (value, axis) => round(value + contexts[position][axis])
      ));
      const vocab = 32;
      const logits = residual.map((row) => vector(vocab, (word) => round(
        row.reduce((sum, value, axis) =>
          sum + value * Math.sin((axis + 1) * (word + 2) * 0.19), 0
        ) / Math.sqrt(dim)
      )));
      const targets = ids.slice(1).map((id) => id % vocab);
      const tokenLosses = logits.slice(0, -1).map((row, position) => {
        const probabilities = softmax(row);
        return -Math.log(Math.max(probabilities[targets[position]], 1e-9));
      });
      const loss = tokenLosses.length
        ? tokenLosses.reduce((sum, value) => sum + value, 0) / tokenLosses.length
        : 0;

      return {
        tokens, ids, dim, heads, headDim, causal, embeddings, q, k, v,
        scores: firstScores, masked: firstMasked, probabilities: firstProb,
        contexts, residual, logits, targets, loss,
      };
    }

    function renderStages() {
      const container = $("[data-stages]");
      container.replaceChildren(...STAGES.map((stage, index) => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = String(index + 1);
        button.title = stage.name;
        button.setAttribute("aria-label", `${index + 1}. ${stage.name}`);
        button.className = index === state.stage ? "is-current" : index < state.stage ? "is-done" : "";
        button.addEventListener("click", () => setStage(index));
        return button;
      }));
      $("[data-prev]").disabled = state.stage === 0;
      $("[data-next]").disabled = state.stage === STAGES.length - 1;
    }

    function shapeRows(model) {
      const shape = (label, value, activeAt) => ({ label, value, active: state.stage >= activeAt });
      return [
        shape("token ids", `[1, ${model.tokens.length}]`, 0),
        shape("embedding x", `[1, ${model.tokens.length}, ${model.dim}]`, 1),
        shape("Q / K / V", `[1, ${model.heads}, ${model.tokens.length}, ${model.headDim}]`, 2),
        shape("attention", `[1, ${model.heads}, ${model.tokens.length}, ${model.tokens.length}]`, 3),
        shape("context + residual", `[1, ${model.tokens.length}, ${model.dim}]`, 5),
        shape("logits", `[1, ${model.tokens.length}, 32]`, 6),
      ];
    }

    function renderShapes(model) {
      $("[data-shapes]").innerHTML = shapeRows(model).map((item) =>
        `<div class="${item.active ? "is-active" : ""}"><span>${item.label}</span><strong>${item.value}</strong></div>`
      ).join("");
    }

    function renderHeatmap(model) {
      const isProbability = state.matrix === "prob";
      const matrix = isProbability ? model.probabilities : model.masked;
      const finite = matrix.flat().filter(Number.isFinite);
      const min = Math.min(...finite);
      const max = Math.max(...finite);
      const range = max - min || 1;
      const cells = [];
      const corner = document.createElement("div");
      corner.className = "lfs-heatmap__axis";
      corner.textContent = "Q↓ K→";
      corner.setAttribute("role", "columnheader");
      cells.push(corner);

      model.tokens.forEach((token) => {
        const header = document.createElement("div");
        header.className = "lfs-heatmap__axis";
        header.textContent = token;
        header.title = token;
        header.setAttribute("role", "columnheader");
        cells.push(header);
      });

      matrix.forEach((row, rowIndex) => {
        const header = document.createElement("div");
        header.className = "lfs-heatmap__axis";
        header.textContent = model.tokens[rowIndex];
        header.title = model.tokens[rowIndex];
        header.setAttribute("role", "rowheader");
        cells.push(header);
        row.forEach((value, columnIndex) => {
          const cell = document.createElement("div");
          const masked = !Number.isFinite(value);
          const intensity = masked ? 0 : (value - min) / range;
          cell.className = `lfs-heatmap__cell${masked ? " is-masked" : ""}`;
          cell.style.setProperty("--heat", String(intensity));
          cell.style.setProperty("--cell-color", `hsl(217 82% ${96 - intensity * 53}%)`);
          cell.style.setProperty("--cell-ink", `hsl(218 35% ${20 + intensity * 70}%)`);
          cell.textContent = masked ? "×" : isProbability ? value.toFixed(2) : value.toFixed(2);
          cell.setAttribute("role", "cell");
          cell.setAttribute("aria-label", `${model.tokens[rowIndex]} 对 ${model.tokens[columnIndex]}：${masked ? "被遮罩" : value.toFixed(3)}`);
          cells.push(cell);
        });
      });
      const heatmap = $("[data-heatmap]");
      heatmap.style.setProperty("--matrix-size", String(model.tokens.length + 1));
      heatmap.replaceChildren(...cells);
      $("[data-matrix-mode]").textContent = isProbability ? "mask 后概率" : "缩放后的 score";
      lab.querySelectorAll("[data-matrix]").forEach((button) => {
        button.classList.toggle("is-active", button.dataset.matrix === state.matrix);
        button.setAttribute("aria-pressed", String(button.dataset.matrix === state.matrix));
      });
    }

    function norm(row) {
      return Math.sqrt(row.reduce((sum, value) => sum + value * value, 0));
    }

    function renderFlow(model) {
      const last = model.tokens.length - 1;
      const flows = [
        { label: "embedding x", value: norm(model.embeddings[last]), stage: 1 },
        { label: "attention context", value: norm(model.contexts[last]), stage: 5 },
        { label: "x + context", value: norm(model.residual[last]), stage: 5 },
        { label: "output head", value: Math.max(...model.logits[last].map(Math.abs)), stage: 6 },
      ];
      const max = Math.max(...flows.map((item) => item.value), 0.001);
      $("[data-flow]").innerHTML = flows.map((item, index) => `
        <div class="${state.stage >= item.stage ? "is-active" : ""}">
          <span>${index ? "→" : ""}</span><b>${item.label}</b>
          <i style="--bar:${Math.max(8, item.value / max * 100).toFixed(1)}%"></i>
          <em>L2 ${item.value.toFixed(2)}</em>
        </div>`).join("");

      const selected = state.stage >= 5 ? model.residual[last] : model.embeddings[last];
      $("[data-vector]").innerHTML = selected.map((value, index) =>
        `<span style="--value:${Math.min(100, Math.abs(value) * 70).toFixed(1)}%;--sign:${value >= 0 ? 1 : 0}" title="d${index}: ${value.toFixed(3)}"><i></i><small>d${index}</small></span>`
      ).join("");

      const top = model.logits[last]
        .map((value, id) => ({ id, value }))
        .sort((a, b) => b.value - a.value)
        .slice(0, 5);
      $("[data-logits]").innerHTML = state.stage >= 6
        ? top.map((item) => `<li><span>token ${item.id}</span><i style="--logit:${Math.max(6, (item.value - top.at(-1).value + 0.2) / (top[0].value - top.at(-1).value + 0.2) * 100).toFixed(1)}%"></i><b>${item.value.toFixed(2)}</b></li>`).join("")
        : `<li class="is-placeholder">到第 7 步查看最后位置的 top logits</li>`;
    }

    function render(announce = false) {
      const model = state.model;
      $("[data-d-value]").textContent = String(model.dim);
      $("[data-step-name]").textContent = STAGES[state.stage].name;
      $("[data-token-count]").textContent = String(model.tokens.length);
      $("[data-head-dim]").textContent = String(model.headDim);
      $("[data-loss]").textContent = state.stage >= 6 ? model.loss.toFixed(3) : "—";
      renderStages();
      renderShapes(model);
      renderHeatmap(model);
      renderFlow(model);
      lab.dataset.stage = STAGES[state.stage].key;
      if (!reducedMotion) lab.classList.add("is-updating");
      window.setTimeout(() => lab.classList.remove("is-updating"), reducedMotion ? 0 : 180);
      if (announce) {
        $("[data-live]").textContent = `第 ${state.stage + 1} 步：${STAGES[state.stage].name}。T=${model.tokens.length}，D=${model.dim}，H=${model.heads}，每头 ${model.headDim} 维。`;
      }
    }

    function setStage(next) {
      state.stage = Math.max(0, Math.min(STAGES.length - 1, next));
      render(true);
    }

    function apply() {
      state.model = buildModel();
      render(true);
    }

    $("[data-controls]").addEventListener("submit", (event) => {
      event.preventDefault();
      apply();
    });
    $("[data-dim]").addEventListener("input", () => {
      $("[data-d-value]").textContent = $("[data-dim]").value;
    });
    $("[data-prev]").addEventListener("click", () => setStage(state.stage - 1));
    $("[data-next]").addEventListener("click", () => setStage(state.stage + 1));
    lab.querySelectorAll("[data-matrix]").forEach((button) => button.addEventListener("click", () => {
      state.matrix = button.dataset.matrix;
      renderHeatmap(state.model);
      $("[data-live]").textContent = state.matrix === "prob" ? "显示 mask 后的 attention probability。" : "显示缩放并遮罩后的 attention score。";
    }));
    lab.addEventListener("keydown", (event) => {
      if (["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) return;
      const actions = {
        ArrowLeft: () => setStage(state.stage - 1),
        ArrowRight: () => setStage(state.stage + 1),
        Home: () => setStage(0),
        End: () => setStage(STAGES.length - 1),
      };
      if (actions[event.key]) {
        event.preventDefault();
        actions[event.key]();
      }
    });

    state.model = buildModel();
    render(false);
  }
})();
