(() => {
  const requests = [
    {
      id: "A",
      name: "解释 Radix Cache",
      tokens: ["<sys>", "你是", "代码", "助手", "解释", "Radix", "Cache"],
    },
    {
      id: "B",
      name: "解释 Chunked Prefill",
      tokens: ["<sys>", "你是", "代码", "助手", "解释", "Chunk", "Prefill"],
    },
    {
      id: "C",
      name: "解释张量并行",
      tokens: ["<sys>", "你是", "系统", "助手", "解释", "张量", "并行"],
    },
    {
      id: "D",
      name: "解释 Decode Attention",
      tokens: ["<sys>", "你是", "代码", "助手", "解释", "Decode", "Attention"],
    },
  ];

  const escapeHtml = (value) =>
    String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");

  function makeNode(token = "root") {
    return { token, children: new Map(), requestIds: new Set() };
  }

  function buildTrie(entries) {
    const root = makeNode();
    entries.forEach((entry) => {
      let node = root;
      node.requestIds.add(entry.id);
      entry.tokens.forEach((token) => {
        if (!node.children.has(token)) node.children.set(token, makeNode(token));
        node = node.children.get(token);
        node.requestIds.add(entry.id);
      });
    });
    return root;
  }

  function trieSize(node) {
    let size = 0;
    node.children.forEach((child) => {
      size += 1 + trieSize(child);
    });
    return size;
  }

  function matchLength(root, tokens) {
    let node = root;
    let matched = 0;
    for (const token of tokens) {
      const child = node.children.get(token);
      if (!child) break;
      matched += 1;
      node = child;
    }
    return matched;
  }

  function compactChildren(node) {
    return [...node.children.values()].map((start) => {
      const tokens = [start.token];
      let cursor = start;
      while (cursor.children.size === 1) {
        const next = [...cursor.children.values()][0];
        if (next.requestIds.size !== cursor.requestIds.size) break;
        tokens.push(next.token);
        cursor = next;
      }
      return { tokens, node: cursor, shared: cursor.requestIds.size > 1 };
    });
  }

  function treeMarkup(node) {
    const branches = compactChildren(node);
    if (!branches.length) return "";
    return `<ul>${branches
      .map(
        ({ tokens, node: endNode, shared }) => `<li>
          <span class="${shared ? "is-shared" : ""}">${tokens
            .map(escapeHtml)
          .join(" · ")}<small>${endNode.requestIds.size} paths</small></span>
          ${treeMarkup(endNode)}
        </li>`,
      )
      .join("")}</ul>`;
  }

  function buildTimeline(results, overlap) {
    const cpuTasks = [];
    const gpuTasks = [];
    let cpuCursor = 0;
    let gpuCursor = 0;

    results.forEach((result) => {
      const cpuStart = overlap ? cpuCursor : Math.max(cpuCursor, gpuCursor);
      const cpuDuration = 0.65 + result.chunks.length * 0.06;
      const cpuEnd = cpuStart + cpuDuration;
      cpuTasks.push({
        start: cpuStart,
        end: cpuEnd,
        label: `调度 ${result.id}`,
        type: "cpu",
      });
      cpuCursor = cpuEnd;

      let gpuStart = Math.max(gpuCursor, cpuEnd);
      result.chunks.forEach((chunk, index) => {
        const end = gpuStart + 0.45 + chunk.length * 0.32;
        gpuTasks.push({
          start: gpuStart,
          end,
          label: `${result.id} · P${index + 1}`,
          type: "prefill",
        });
        gpuStart = end;
      });
      const decodeEnd = gpuStart + 0.72;
      gpuTasks.push({
        start: gpuStart,
        end: decodeEnd,
        label: `${result.id} · D`,
        type: "decode",
      });
      gpuCursor = decodeEnd;
      if (!overlap) cpuCursor = gpuCursor;
    });

    return { cpuTasks, gpuTasks, total: Math.max(cpuCursor, gpuCursor) };
  }

  function simulate(chunkSize, capacity, overlap) {
    let active = [];
    let accessTick = 0;
    let totalHits = 0;
    let totalPrefill = 0;
    let totalEvicted = 0;
    const results = [];
    const evictions = [];

    requests.forEach((request) => {
      let trie = buildTrie(active);
      const hit = Math.min(matchLength(trie, request.tokens), request.tokens.length - 1);
      const pending = request.tokens.slice(hit);
      const chunks = [];
      for (let offset = 0; offset < pending.length; offset += chunkSize) {
        chunks.push(pending.slice(offset, offset + chunkSize));
      }

      totalHits += hit;
      totalPrefill += pending.length;
      accessTick += 1;
      active.forEach((entry) => {
        if (entry.tokens.slice(0, hit).every((token, index) => token === request.tokens[index])) {
          entry.lastUsed = accessTick;
        }
      });
      active.push({ ...request, lastUsed: accessTick });
      trie = buildTrie(active);

      const requestEvictions = [];
      while (trieSize(trie) > capacity && active.length > 1) {
        const candidates = active
          .filter((entry) => entry.id !== request.id)
          .sort((a, b) => a.lastUsed - b.lastUsed || a.id.localeCompare(b.id));
        const victim = candidates[0] || active[0];
        const before = trieSize(trie);
        active = active.filter((entry) => entry.id !== victim.id);
        trie = buildTrie(active);
        const freed = before - trieSize(trie);
        totalEvicted += freed;
        requestEvictions.push(victim.id);
        evictions.push({ after: request.id, victim: victim.id, freed });
      }

      results.push({ ...request, hit, chunks, evictions: requestEvictions });
    });

    const trie = buildTrie(active);
    return {
      results,
      evictions,
      trie,
      usage: trieSize(trie),
      totalHits,
      totalPrefill,
      totalEvicted,
      timeline: buildTimeline(results, overlap),
    };
  }

  function renderRequests(container, results) {
    container.innerHTML = results
      .map((result) => {
        const tokenMarkup = result.tokens
          .map(
            (token, index) =>
              `<span class="${index < result.hit ? "is-hit" : "is-miss"}">${escapeHtml(
                token,
              )}</span>`,
          )
          .join("");
        const chunkMarkup = result.chunks
          .map(
            (chunk, index) =>
              `<span><b>P${index + 1}</b>${chunk.map(escapeHtml).join(" · ")}</span>`,
          )
          .join("");
        const evicted = result.evictions.length
          ? `<em>触发淘汰 ${result.evictions.join(", ")}</em>`
          : "";
        return `<article>
          <header><b>${result.id}</b><span>${escapeHtml(result.name)}</span><small>hit ${
            result.hit
          } / prefill ${result.tokens.length - result.hit}</small></header>
          <div class="msgl-token-row">${tokenMarkup}</div>
          <div class="msgl-chunk-row">${chunkMarkup}<i>→ D</i>${evicted}</div>
        </article>`;
      })
      .join("");
  }

  function renderTimeline(container, timeline) {
    const makeLane = (label, tasks) => `<div class="msgl-lane">
      <b>${label}</b>
      <div class="msgl-lane-track">${tasks
        .map((task) => {
          const left = (task.start / timeline.total) * 100;
          const width = ((task.end - task.start) / timeline.total) * 100;
          return `<span class="is-${task.type}" style="left:${left}%;width:${width}%" title="${escapeHtml(
            `${task.label}: ${task.start.toFixed(1)}–${task.end.toFixed(1)} ms`,
          )}">${escapeHtml(task.label)}</span>`;
        })
        .join("")}</div>
    </div>`;
    container.innerHTML = `${makeLane("CPU", timeline.cpuTasks)}${makeLane(
      "GPU",
      timeline.gpuTasks,
    )}<div class="msgl-axis"><span>0 ms</span><span>${timeline.total.toFixed(1)} ms</span></div>`;
  }

  function initGuide(guide) {
    const lab = guide.querySelector("[data-msgl-lab]");
    if (!lab || lab.dataset.ready === "true") return;
    lab.dataset.ready = "true";
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      lab.dataset.reducedMotion = "true";
    }

    const form = lab.querySelector("[data-controls]");
    const chunkInput = lab.querySelector("[data-chunk]");
    const capacityInput = lab.querySelector("[data-capacity]");
    const overlapInput = lab.querySelector("[data-overlap]");
    const chunkValue = lab.querySelector("[data-chunk-value]");
    const capacityValue = lab.querySelector("[data-capacity-value]");

    const render = () => {
      const chunkSize = Number(chunkInput.value);
      const capacity = Number(capacityInput.value);
      const overlap = overlapInput.checked;
      const model = simulate(chunkSize, capacity, overlap);

      chunkValue.value = String(chunkSize);
      capacityValue.value = String(capacity);
      lab.querySelector("[data-hit-total]").textContent = String(model.totalHits);
      lab.querySelector("[data-prefill-total]").textContent = String(model.totalPrefill);
      lab.querySelector("[data-evict-total]").textContent = String(model.totalEvicted);
      lab.querySelector("[data-total-time]").textContent = model.timeline.total.toFixed(1);
      lab.querySelector("[data-cache-usage]").textContent = `${model.usage} / ${capacity} tokens`;
      lab.querySelector("[data-overlap-state]").textContent = overlap
        ? "overlap on"
        : "overlap off";

      renderRequests(lab.querySelector("[data-request-list]"), model.results);
      const tree = lab.querySelector("[data-tree]");
      tree.innerHTML = `<div class="msgl-tree-root">root</div>${treeMarkup(model.trie)}`;
      const evictionList = lab.querySelector("[data-eviction-list]");
      evictionList.innerHTML = model.evictions.length
        ? model.evictions
            .map(
              (event) =>
                `<li>请求 ${event.after} 后淘汰 ${event.victim} 的叶路径，释放 ${event.freed} token</li>`,
            )
            .join("")
        : "<li>当前容量未触发淘汰</li>";
      renderTimeline(lab.querySelector("[data-timeline]"), model.timeline);

      lab.querySelector(
        "[data-live]",
      ).textContent = `已重新调度：chunk ${chunkSize}，缓存 ${capacity} token，前缀命中 ${model.totalHits} token，淘汰 ${model.totalEvicted} token，模型时间线 ${model.timeline.total.toFixed(
        1,
      )} 毫秒。`;
    };

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      render();
    });
    [chunkInput, capacityInput, overlapInput].forEach((control) => {
      control.addEventListener("input", render);
      control.addEventListener("change", render);
    });
    render();
  }

  const init = () =>
    document.querySelectorAll('[data-guide="mini-sglang"]').forEach(initGuide);

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
