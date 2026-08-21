(() => {
  const roots = document.querySelectorAll('[data-guide="micrograd"]');
  roots.forEach((root, index) => initLab(root, index));

  function initLab(root, rootIndex) {
    const lab = root.querySelector("[data-mg-lab]");
    if (!lab) return;

    const query = (selector) => lab.querySelector(selector);
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const arrowId = `mg-arrow-${rootIndex}`;
    let graph;
    let reverseOrder = [];
    let cursor = 0;
    let currentId = null;
    let executed = new Set();
    let playTimer = null;
    let lastFormula = "点击“下一步”给输出梯度置 1。";

    function numberFrom(selector, fallback) {
      const input = query(selector);
      const raw = Number(input.value);
      const value = Number.isFinite(raw) ? Math.max(-10, Math.min(10, raw)) : fallback;
      input.value = String(value);
      return value;
    }

    function makeNode(id, label, value, kind = "op") {
      return {
        id,
        label,
        value,
        grad: 0,
        kind,
        op: "",
        parents: [],
        position: { x: 0, y: 0 },
        back() {
          return [];
        },
      };
    }

    function attachOperation(out, op, links) {
      out.op = op;
      out.parents = links.map((link) => link.node);
      out.links = links;
      out.back = () => links.map((link) => {
        const local = link.local();
        const delta = out.grad * local;
        const before = link.node.grad;
        link.node.grad += delta;
        return { parent: link.node, local, delta, before, after: link.node.grad };
      });
      return out;
    }

    function add(a, b, id, label) {
      return attachOperation(makeNode(id, label, a.value + b.value), "+", [
        { node: a, local: () => 1 },
        { node: b, local: () => 1 },
      ]);
    }

    function multiply(a, b, id, label) {
      return attachOperation(makeNode(id, label, a.value * b.value), "×", [
        { node: a, local: () => b.value },
        { node: b, local: () => a.value },
      ]);
    }

    function square(a, id, label) {
      return attachOperation(makeNode(id, label, a.value ** 2), "²", [
        { node: a, local: () => 2 * a.value },
      ]);
    }

    function relu(a, id, label) {
      return attachOperation(makeNode(id, label, Math.max(0, a.value)), "ReLU", [
        { node: a, local: () => (a.value > 0 ? 1 : 0) },
      ]);
    }

    function buildGraph() {
      const expression = query("[data-mg-expression]").value;
      const x = makeNode("x", "x", numberFrom('[data-mg-input="x"]', 2), "leaf");
      const w = makeNode("w", "w", numberFrom('[data-mg-input="w"]', -3), "leaf");
      const b = makeNode("b", "b", numberFrom('[data-mg-input="b"]', 1), "leaf");
      const product = multiply(x, w, "product", "x × w");
      let output;
      let nodes;

      if (expression === "relu") {
        const affine = add(product, b, "affine", "xw + b");
        output = relu(affine, "output", "y = ReLU");
        nodes = [x, w, b, product, affine, output];
        Object.assign(x.position, { x: 95, y: 75 });
        Object.assign(w.position, { x: 95, y: 225 });
        Object.assign(b.position, { x: 370, y: 350 });
        Object.assign(product.position, { x: 370, y: 150 });
        Object.assign(affine.position, { x: 625, y: 225 });
        Object.assign(output.position, { x: 870, y: 225 });
      } else if (expression === "shared") {
        const squared = square(x, "square", "x²");
        const merged = add(product, squared, "merged", "xw + x²");
        output = add(merged, b, "output", "y");
        nodes = [x, w, b, product, squared, merged, output];
        Object.assign(x.position, { x: 85, y: 130 });
        Object.assign(w.position, { x: 85, y: 315 });
        Object.assign(b.position, { x: 590, y: 380 });
        Object.assign(product.position, { x: 330, y: 270 });
        Object.assign(squared.position, { x: 330, y: 75 });
        Object.assign(merged.position, { x: 590, y: 170 });
        Object.assign(output.position, { x: 850, y: 245 });
      } else {
        output = add(product, b, "output", "y");
        nodes = [x, w, b, product, output];
        Object.assign(x.position, { x: 95, y: 90 });
        Object.assign(w.position, { x: 95, y: 290 });
        Object.assign(b.position, { x: 420, y: 370 });
        Object.assign(product.position, { x: 420, y: 170 });
        Object.assign(output.position, { x: 790, y: 250 });
      }

      return { expression, nodes, output, inputs: { x, w, b } };
    }

    function topologicalOrder(output) {
      const visited = new Set();
      const order = [];
      function visit(node) {
        if (visited.has(node.id)) return;
        visited.add(node.id);
        node.parents.forEach(visit);
        order.push(node);
      }
      visit(output);
      return order;
    }

    function reset(announceReset = true) {
      stopPlaying();
      graph = buildGraph();
      reverseOrder = topologicalOrder(graph.output).reverse();
      cursor = 0;
      currentId = null;
      executed = new Set();
      lastFormula = "点击“下一步”给输出梯度置 1。";
      render();
      if (announceReset) {
        announce(`计算图已重建，输出 y 为 ${format(graph.output.value)}，共 ${reverseOrder.length} 个反向节点。`);
      }
    }

    function step() {
      if (cursor >= reverseOrder.length) {
        stopPlaying();
        announce("反向传播已完成；解析梯度可与有限差分对照。 ");
        return;
      }

      const node = reverseOrder[cursor];
      if (cursor === 0) node.grad = 1;
      const contributions = node.back();
      currentId = node.id;
      executed.add(node.id);
      cursor += 1;

      const seed = cursor === 1 ? `${node.label}.grad = 1；` : "";
      if (contributions.length) {
        const details = contributions.map((item) =>
          `${item.parent.label}.grad += ${format(node.grad)} × ${format(item.local)} = ${signed(item.delta)}（${format(item.before)} → ${format(item.after)}）`,
        );
        lastFormula = `${seed}${details.join("；")}`;
      } else {
        lastFormula = `${seed}${node.label} 是叶子节点，没有需要继续执行的反向闭包。`;
      }

      render();
      const completion = cursor >= reverseOrder.length ? "反向传播完成。" : `下一节点是 ${reverseOrder[cursor].label}。`;
      announce(`已执行 ${node.label} 的反向闭包。${completion}`);
      if (cursor >= reverseOrder.length) stopPlaying();
    }

    function togglePlay() {
      if (playTimer) {
        stopPlaying();
        announce("已暂停反向传播。 ");
        return;
      }
      if (cursor >= reverseOrder.length) reset(false);
      query("[data-mg-play]").textContent = "暂停";
      step();
      if (cursor < reverseOrder.length) {
        playTimer = window.setInterval(step, reducedMotion ? 1100 : 760);
      }
    }

    function stopPlaying() {
      if (playTimer) window.clearInterval(playTimer);
      playTimer = null;
      const button = query("[data-mg-play]");
      if (button) button.textContent = "连续播放";
    }

    function evaluate(expression, values) {
      const product = values.x * values.w;
      if (expression === "relu") return Math.max(0, product + values.b);
      if (expression === "shared") return product + values.x ** 2 + values.b;
      return product + values.b;
    }

    function numericGradient(name) {
      const epsilon = 1e-4;
      const values = {
        x: graph.inputs.x.value,
        w: graph.inputs.w.value,
        b: graph.inputs.b.value,
      };
      const plus = { ...values, [name]: values[name] + epsilon };
      const minus = { ...values, [name]: values[name] - epsilon };
      return (evaluate(graph.expression, plus) - evaluate(graph.expression, minus)) / (2 * epsilon);
    }

    function render() {
      query("[data-mg-phase]").textContent = cursor === 0
        ? "等待反向"
        : cursor >= reverseOrder.length
          ? "已完成"
          : "反向传播";
      query("[data-mg-current]").textContent = currentId
        ? graph.nodes.find((node) => node.id === currentId).label
        : "—";
      query("[data-mg-progress]").textContent = `${cursor} / ${reverseOrder.length}`;
      query("[data-mg-output]").textContent = format(graph.output.value);
      query("[data-mg-formula]").textContent = lastFormula;
      query("[data-mg-step]").disabled = cursor >= reverseOrder.length;
      renderGraph();
      renderTopology();
      renderCheck();
    }

    function renderGraph() {
      const edgeMarkup = graph.nodes.flatMap((node) => (node.links || []).map((link, index) => {
        const parent = link.node;
        const x1 = node.position.x - 73;
        const x2 = parent.position.x + 73;
        const y1 = node.position.y;
        const y2 = parent.position.y;
        const labelX = x2 + (x1 - x2) * 0.5;
        const labelY = y2 + (y1 - y2) * 0.5 - (index ? 10 : -12);
        const active = node.id === currentId ? " is-active" : "";
        return `<g class="mg-svg-edge${active}">
          <path d="M ${x1} ${y1} C ${x1 - 65} ${y1}, ${x2 + 65} ${y2}, ${x2} ${y2}" marker-end="url(#${arrowId})"></path>
          <rect x="${labelX - 39}" y="${labelY - 11}" width="78" height="22" rx="7"></rect>
          <text x="${labelX}" y="${labelY + 4}">local ${format(link.local())}</text>
        </g>`;
      })).join("");

      const nodeMarkup = graph.nodes.map((node) => {
        const classes = ["mg-svg-node", `is-${node.kind}`];
        if (node.id === currentId) classes.push("is-current");
        else if (executed.has(node.id)) classes.push("is-done");
        return `<g class="${classes.join(" ")}" transform="translate(${node.position.x - 73} ${node.position.y - 37})">
          <rect width="146" height="74" rx="14"></rect>
          <text class="mg-svg-node__label" x="73" y="21">${escapeHtml(node.label)}</text>
          <text x="73" y="43">value ${format(node.value)}</text>
          <text class="mg-svg-node__grad" x="73" y="62">grad ${format(node.grad)}</text>
        </g>`;
      }).join("");

      query("[data-mg-graph]").innerHTML = `<svg viewBox="0 0 960 460" role="img" aria-label="${escapeHtml(graphDescription())}" preserveAspectRatio="xMidYMid meet">
        <defs><marker id="${arrowId}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z"></path></marker></defs>
        ${edgeMarkup}${nodeMarkup}
      </svg>`;
    }

    function renderTopology() {
      query("[data-mg-topology]").innerHTML = reverseOrder.map((node, index) => {
        const state = index < cursor ? "done" : index === cursor ? "next" : "waiting";
        return `<li data-state="${state}"${node.id === currentId ? ' data-current="true"' : ""}><span>${index + 1}</span>${escapeHtml(node.label)}</li>`;
      }).join("");
    }

    function renderCheck() {
      const complete = cursor >= reverseOrder.length;
      query("[data-mg-check]").innerHTML = ["x", "w", "b"].map((name) => {
        const analytic = graph.inputs[name].grad;
        const numeric = numericGradient(name);
        const difference = Math.abs(analytic - numeric);
        return `<tr${complete && difference < 1e-6 ? ' data-match="true"' : ""}>
          <th>${name}</th><td>${format(analytic)}</td><td>${format(numeric)}</td><td>${complete ? scientific(difference) : "待完成"}</td>
        </tr>`;
      }).join("");
    }

    function graphDescription() {
      return graph.nodes.map((node) => `${node.label}，值 ${format(node.value)}，梯度 ${format(node.grad)}`).join("；");
    }

    function format(value) {
      if (Math.abs(value) < 1e-10) return "0";
      if (Number.isInteger(value)) return String(value);
      return Number(value.toFixed(4)).toString();
    }

    function signed(value) {
      return value >= 0 ? `+${format(value)}` : format(value);
    }

    function scientific(value) {
      if (value === 0) return "0";
      return value < 1e-4 ? value.toExponential(2) : format(value);
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function announce(message) {
      query("[data-mg-live]").textContent = message;
    }

    query("[data-mg-reset]").addEventListener("click", () => reset());
    query("[data-mg-step]").addEventListener("click", step);
    query("[data-mg-play]").addEventListener("click", togglePlay);
    query("[data-mg-expression]").addEventListener("change", () => reset());
    lab.querySelectorAll("[data-mg-input]").forEach((input) => {
      input.addEventListener("change", () => reset());
    });
    lab.addEventListener("keydown", (event) => {
      if (["INPUT", "SELECT", "BUTTON"].includes(event.target.tagName)) return;
      if (event.key === "ArrowRight") {
        event.preventDefault();
        step();
      } else if (event.key === " ") {
        event.preventDefault();
        togglePlay();
      } else if (event.key.toLowerCase() === "r") {
        event.preventDefault();
        reset();
      }
    });

    reset(false);
  }
})();
