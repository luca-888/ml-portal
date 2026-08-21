(() => {
  const roots = document.querySelectorAll('[data-guide="nano-vllm"]');
  roots.forEach((root) => initLab(root));

  function initLab(root) {
    const lab = root.querySelector("[data-nv-lab]");
    if (!lab) return;

    const query = (selector) => lab.querySelector(selector);
    const queryAll = (selector) => [...lab.querySelectorAll(selector)];
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const colors = { A: "a", B: "b", C: "c" };
    let state;
    let playTimer = null;

    lab.tabIndex = 0;

    const clamp = (value, min, max) => Math.min(max, Math.max(min, Number(value) || min));
    const byId = (id) => state.requests.find((request) => request.id === id);
    const freePages = () => state.pages.filter((page) => page.requestId === null);

    function readConfiguration() {
      const pageCount = clamp(query("[data-nv-pages]").value, 3, 24);
      const pageSize = clamp(query("[data-nv-page-size]").value, 2, 16);
      const budget = clamp(query("[data-nv-budget]").value, 1, 32);
      query("[data-nv-pages]").value = pageCount;
      query("[data-nv-page-size]").value = pageSize;
      query("[data-nv-budget]").value = budget;

      const requests = queryAll("[data-nv-request]").map((fieldset) => {
        const id = fieldset.dataset.nvRequest;
        const arrival = clamp(fieldset.querySelector('[data-field="arrival"]').value, 0, 20);
        const prompt = clamp(fieldset.querySelector('[data-field="prompt"]').value, 1, 64);
        const output = clamp(fieldset.querySelector('[data-field="output"]').value, 1, 32);
        fieldset.querySelector('[data-field="arrival"]').value = arrival;
        fieldset.querySelector('[data-field="prompt"]').value = prompt;
        fieldset.querySelector('[data-field="output"]').value = output;
        return {
          id,
          arrival,
          prompt,
          output,
          promptDone: 0,
          outputDone: 0,
          cachedTokens: 0,
          status: "future",
          pages: [],
          preemptions: 0,
        };
      });

      return { pageCount, pageSize, budget, requests };
    }

    function reset() {
      stopPlaying();
      const config = readConfiguration();
      state = {
        clock: 0,
        phase: "等待开始",
        scheduled: [],
        waiting: [],
        running: [],
        finished: [],
        events: [],
        pageSize: config.pageSize,
        budget: config.budget,
        pages: Array.from({ length: config.pageCount }, (_, id) => ({
          id,
          requestId: null,
          logical: null,
          used: 0,
        })),
        requests: config.requests,
      };
      addEvent("参数已应用；step 0 等待调度。", "reset");
      render();
      announce(`模拟器已重置，共 ${config.pageCount} 个物理页，每页 ${config.pageSize} 个 token。`);
    }

    function addEvent(message, type = "normal") {
      state.events.unshift({ step: state.clock, message, type });
      state.events = state.events.slice(0, 14);
    }

    function announce(message) {
      query("[data-nv-live]").textContent = message;
    }

    function loadArrivals() {
      state.requests
        .filter((request) => request.status === "future" && request.arrival <= state.clock)
        .sort((a, b) => a.arrival - b.arrival || a.id.localeCompare(b.id))
        .forEach((request) => {
          request.status = "waiting";
          state.waiting.push(request.id);
          addEvent(`${request.id} 到达，进入 WAITING。`, "arrival");
        });
    }

    function allocatePages(request, count) {
      const candidates = freePages().slice(0, count);
      if (candidates.length < count) return false;
      candidates.forEach((page) => {
        page.requestId = request.id;
        page.logical = request.pages.length;
        page.used = 0;
        request.pages.push(page.id);
      });
      updatePageUsage(request);
      return true;
    }

    function releasePages(request) {
      request.pages.forEach((pageId) => {
        const page = state.pages[pageId];
        page.requestId = null;
        page.logical = null;
        page.used = 0;
      });
      request.pages = [];
    }

    function updatePageUsage(request) {
      request.pages.forEach((pageId, logical) => {
        const page = state.pages[pageId];
        page.used = Math.max(0, Math.min(state.pageSize, request.cachedTokens - logical * state.pageSize));
      });
    }

    function finish(request) {
      state.waiting = state.waiting.filter((id) => id !== request.id);
      state.running = state.running.filter((id) => id !== request.id);
      request.status = "finished";
      releasePages(request);
      if (!state.finished.includes(request.id)) state.finished.push(request.id);
      addEvent(`${request.id} 生成 ${request.outputDone}/${request.output}，释放全部 pages。`, "finish");
    }

    function preempt(request) {
      state.running = state.running.filter((id) => id !== request.id);
      releasePages(request);
      request.status = "waiting";
      request.promptDone = 0;
      request.outputDone = 0;
      request.cachedTokens = 0;
      request.preemptions += 1;
      state.waiting = state.waiting.filter((id) => id !== request.id);
      state.waiting.unshift(request.id);
      addEvent(`${request.id} 被抢占：释放 pages，回到 WAITING 重做 prefill。`, "preempt");
    }

    function runPrefill() {
      let remainingBudget = state.budget;
      const scheduled = [];

      while (state.waiting.length && remainingBudget > 0) {
        const request = byId(state.waiting[0]);
        const requiredPages = Math.ceil(request.prompt / state.pageSize);

        if (request.pages.length === 0) {
          if (freePages().length < requiredPages) break;
          allocatePages(request, requiredPages);
          addEvent(`${request.id} 分配 ${requiredPages} 个 prompt pages。`, "allocation");
        }

        const promptRemaining = request.prompt - request.promptDone;
        if (scheduled.length > 0 && promptRemaining > remainingBudget) break;

        const tokenCount = Math.min(promptRemaining, remainingBudget);
        request.promptDone += tokenCount;
        request.cachedTokens += tokenCount;
        remainingBudget -= tokenCount;
        scheduled.push({ id: request.id, tokens: tokenCount, phase: "prefill" });
        updatePageUsage(request);

        if (request.promptDone === request.prompt) {
          state.waiting.shift();
          request.status = "running";
          request.outputDone = 1;
          state.running.push(request.id);
          addEvent(`${request.id} prefill 完成并采样第 1 个输出 token。`, "phase");
          if (request.outputDone >= request.output) finish(request);
        } else {
          addEvent(`${request.id} chunked prefill：本步处理 ${tokenCount} 个 token。`, "phase");
          break;
        }
      }

      return scheduled;
    }

    function runDecode() {
      const scheduled = [];
      const resumed = [];
      const maxCandidates = state.running.length;
      let inspected = 0;

      while (state.running.length && inspected < maxCandidates && scheduled.length < state.budget) {
        let request = byId(state.running.shift());
        inspected += 1;
        const requiredPages = Math.ceil((request.cachedTokens + 1) / state.pageSize);

        while (request && request.pages.length < requiredPages) {
          if (freePages().length) {
            allocatePages(request, 1);
            addEvent(`${request.id} decode 扩展 1 个 KV page。`, "allocation");
            break;
          }

          if (state.running.length) {
            const victim = byId(state.running[state.running.length - 1]);
            preempt(victim);
          } else {
            preempt(request);
            request = null;
          }
        }

        if (!request) continue;
        request.cachedTokens += 1;
        request.outputDone += 1;
        updatePageUsage(request);
        scheduled.push({ id: request.id, tokens: 1, phase: "decode" });

        if (request.outputDone >= request.output) {
          finish(request);
        } else {
          resumed.push(request.id);
        }
      }

      state.running = resumed.concat(state.running);
      return scheduled;
    }

    function isComplete() {
      return state.requests.every((request) => request.status === "finished");
    }

    function step() {
      if (isComplete()) {
        stopPlaying();
        announce("所有请求均已完成。修改参数后可重置模拟器。");
        return;
      }

      loadArrivals();
      let scheduled = runPrefill();
      if (scheduled.length) {
        state.phase = "PREFILL";
      } else if (state.running.length) {
        scheduled = runDecode();
        state.phase = scheduled.length ? "DECODE" : "PREEMPT";
      } else if (state.waiting.length) {
        state.phase = "PAGE WAIT";
        addEvent(`${state.waiting[0]} 等待足够的物理 pages。`, "wait");
      } else {
        state.phase = "IDLE";
        addEvent("当前无可运行请求，等待下一次到达。", "wait");
      }

      state.scheduled = scheduled;
      const summary = scheduled.length
        ? scheduled.map((item) => `${item.id}:${item.tokens}`).join(" · ")
        : "—";
      const spokenPhase = state.phase;
      state.clock += 1;
      render();
      announce(`step ${state.clock - 1}，${spokenPhase}，调度 ${summary}。`);

      if (isComplete()) {
        stopPlaying();
        announce(`step ${state.clock - 1} 后所有请求完成。`);
      }
    }

    function togglePlay() {
      if (playTimer) {
        stopPlaying();
        return;
      }
      query("[data-nv-play]").textContent = "暂停";
      step();
      if (!isComplete()) {
        playTimer = window.setInterval(step, reducedMotion ? 1000 : 700);
      }
    }

    function stopPlaying() {
      if (playTimer) window.clearInterval(playTimer);
      playTimer = null;
      const button = query("[data-nv-play]");
      if (button) button.textContent = "连续运行";
    }

    function requestChip(id) {
      const request = byId(id);
      return `<span class="nv-request-chip" data-request="${colors[id]}">
        <b>${id}</b><small>P ${request.promptDone}/${request.prompt} · O ${request.outputDone}/${request.output}</small>
        ${request.preemptions ? `<em>抢占 ${request.preemptions}</em>` : ""}
      </span>`;
    }

    function renderQueue(selector, ids) {
      const target = query(selector);
      target.innerHTML = ids.length ? ids.map(requestChip).join("") : '<span class="nv-empty">空</span>';
    }

    function renderTimeline() {
      query("[data-nv-timeline]").innerHTML = state.requests.map((request) => {
        const statusLabel = {
          future: `t${request.arrival} 到达`,
          waiting: "WAITING",
          running: "RUNNING",
          finished: "FINISHED",
        }[request.status];
        const promptPercent = (request.promptDone / request.prompt) * 100;
        const outputPercent = (request.outputDone / request.output) * 100;
        return `<div class="nv-progress-row" data-request="${colors[request.id]}">
          <div><b>${request.id}</b><span>${statusLabel}</span></div>
          <div class="nv-progress-track">
            <span class="nv-progress-zone"><i style="width:${promptPercent}%"></i><small>P ${request.promptDone}/${request.prompt}</small></span>
            <span class="nv-progress-zone nv-progress-zone--output"><i style="width:${outputPercent}%"></i><small>O ${request.outputDone}/${request.output}</small></span>
          </div>
        </div>`;
      }).join("");
    }

    function renderMemory() {
      query("[data-nv-memory-note]").textContent = `每页 ${state.pageSize} tokens`;
      query("[data-nv-physical]").innerHTML = state.pages.map((page) => {
        const slots = Array.from({ length: state.pageSize }, (_, index) =>
          `<i class="${index < page.used ? "is-used" : ""}"></i>`).join("");
        const owner = page.requestId
          ? `<b>${page.requestId}<small>L${page.logical}</small></b>`
          : "<b>FREE</b>";
        return `<div class="nv-page ${page.requestId ? "is-used" : ""}" ${page.requestId ? `data-request="${colors[page.requestId]}"` : ""}>
          <span>P${page.id}</span>${owner}<div class="nv-page__slots">${slots}</div>
        </div>`;
      }).join("");

      const activeTables = state.requests.filter((request) => request.pages.length);
      query("[data-nv-tables]").innerHTML = activeTables.length
        ? activeTables.map((request) => `<div class="nv-table-row" data-request="${colors[request.id]}">
            <b>${request.id}</b><span class="nv-table-arrow">logical</span>
            ${request.pages.map((pageId, logical) => `<span>L${logical}<i>→</i>P${pageId}</span>`).join("")}
          </div>`).join("")
        : '<div class="nv-empty-table">尚无 block table</div>';
    }

    function renderEvents() {
      query("[data-nv-events]").innerHTML = state.events.map((event) =>
        `<li data-event="${event.type}"><span>t${event.step}</span>${event.message}</li>`).join("");
    }

    function render() {
      const usedCount = state.pages.filter((page) => page.requestId !== null).length;
      const utilization = Math.round((usedCount / state.pages.length) * 100);
      query("[data-nv-clock]").textContent = state.clock;
      query("[data-nv-phase]").textContent = state.phase;
      query("[data-nv-scheduled]").textContent = state.scheduled.length
        ? state.scheduled.map((item) => `${item.id}:${item.tokens}`).join(" · ")
        : "—";
      query("[data-nv-util]").textContent = `${utilization}%`;
      query(".nv-lab__meter").style.setProperty("--util", `${utilization}%`);
      query("[data-nv-wait-count]").textContent = state.waiting.length;
      query("[data-nv-run-count]").textContent = state.running.length;
      query("[data-nv-finish-count]").textContent = state.finished.length;
      renderQueue("[data-nv-waiting]", state.waiting);
      renderQueue("[data-nv-running]", state.running);
      renderQueue("[data-nv-finished]", state.finished);
      renderTimeline();
      renderMemory();
      renderEvents();
    }

    query("[data-nv-step]").addEventListener("click", step);
    query("[data-nv-play]").addEventListener("click", togglePlay);
    query("[data-nv-reset]").addEventListener("click", reset);
    lab.addEventListener("keydown", (event) => {
      const target = event.target;
      const isControl = /^(INPUT|BUTTON|SELECT|TEXTAREA)$/.test(target.tagName);
      if (event.code === "Space" && !isControl) {
        event.preventDefault();
        step();
      }
      if (event.key.toLowerCase() === "r" && !isControl) {
        event.preventDefault();
        reset();
      }
    });

    reset();
  }
})();
