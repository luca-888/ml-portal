(() => {
  document.querySelectorAll('[data-guide="gpu-puzzles"]').forEach(initGpuPuzzlesLab);

  function initGpuPuzzlesLab(root) {
    const lab = root.querySelector("[data-gp-lab]");
    if (!lab) return;

    const get = (selector) => lab.querySelector(selector);
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const controls = {
      shape: get("[data-gp-shape]"),
      width: get("[data-gp-width]"),
      height: get("[data-gp-height]"),
      heightLabel: get("[data-gp-height-label]"),
      blockX: get("[data-gp-block-x]"),
      blockY: get("[data-gp-block-y]"),
      mode: get("[data-gp-mode]"),
      stride: get("[data-gp-stride]"),
      block: get("[data-gp-block]"),
    };
    let config;
    let selectedBlock = 0;
    let stepIndex = 0;
    let playTimer = null;
    let matrixHeight = Number(controls.height.value);
    let matrixBlockY = Number(controls.blockY.value);

    const clamp = (value, min, max) =>
      Math.min(max, Math.max(min, Math.round(Number(value) || min)));

    function readConfig() {
      const vector = controls.shape.value === "vector";
      const width = clamp(controls.width.value, 1, 24);
      const height = vector ? 1 : clamp(controls.height.value, 1, 12);
      const blockX = clamp(controls.blockX.value, 1, 16);
      const blockY = vector ? 1 : clamp(controls.blockY.value, 1, 8);
      const stride = clamp(controls.stride.value, 2, 16);

      controls.width.value = width;
      controls.height.value = height;
      controls.blockX.value = blockX;
      controls.blockY.value = blockY;
      controls.stride.value = stride;

      return {
        vector,
        width,
        height,
        blockX,
        blockY,
        mode: controls.mode.value === "strided" ? "strided" : "contiguous",
        stride,
        gridX: Math.ceil(width / blockX),
        gridY: Math.ceil(height / blockY),
      };
    }

    function blockCoordinates(index) {
      return { bx: index % config.gridX, by: Math.floor(index / config.gridX) };
    }

    function addressFor(linear, mode) {
      return linear * (mode === "strided" ? config.stride : 1);
    }

    function threadsForBlock(index, mode = config.mode) {
      const { bx, by } = blockCoordinates(index);
      const threads = [];
      for (let ty = 0; ty < config.blockY; ty += 1) {
        for (let tx = 0; tx < config.blockX; tx += 1) {
          const lane = ty * config.blockX + tx;
          const gx = bx * config.blockX + tx;
          const gy = by * config.blockY + ty;
          const valid = gx < config.width && gy < config.height;
          const linear = gy * config.width + gx;
          const address = valid ? addressFor(linear, mode) : null;
          threads.push({
            lane,
            warp: Math.floor(lane / 32),
            tx,
            ty,
            gx,
            gy,
            valid,
            linear,
            address,
            segment: valid ? Math.floor(address / 8) : null,
          });
        }
      }
      return threads;
    }

    function totalBlocks() {
      return config.gridX * config.gridY;
    }

    function selectedThreads(mode = config.mode) {
      return threadsForBlock(selectedBlock, mode);
    }

    function syncBlockOptions() {
      const count = totalBlocks();
      const previous = Math.min(selectedBlock, count - 1);
      controls.block.innerHTML = Array.from({ length: count }, (_, index) => {
        const { bx, by } = blockCoordinates(index);
        return `<option value="${index}">block (${bx}, ${by})</option>`;
      }).join("");
      selectedBlock = previous;
      controls.block.value = String(selectedBlock);
    }

    function segmentClass(segment) {
      return segment === null ? "" : String(Math.abs(segment) % 6);
    }

    function renderSummary() {
      const launched = totalBlocks() * config.blockX * config.blockY;
      const valid = config.width * config.height;
      const oob = launched - valid;
      get("[data-gp-summary]").innerHTML = `
        <span><b>数据</b>${config.vector ? config.width : `${config.width} × ${config.height}`}</span>
        <span><b>gridDim</b>${config.gridX} × ${config.gridY}</span>
        <span><b>blockDim</b>${config.blockX} × ${config.blockY}</span>
        <span><b>启动线程</b>${launched}</span>
        <span><b>有效 / 越界</b>${valid} / ${oob}</span>
        <span><b>当前地址</b>${config.mode === "contiguous" ? "linear" : `linear × ${config.stride}`}</span>`;
    }

    function renderGrid() {
      const blocks = [];
      for (let blockIndex = 0; blockIndex < totalBlocks(); blockIndex += 1) {
        const { bx, by } = blockCoordinates(blockIndex);
        const cells = threadsForBlock(blockIndex).map((thread) => {
          const active = blockIndex === selectedBlock && thread.lane === stepIndex;
          const coordinate = config.vector ? `g${thread.gx}` : `g${thread.gx},${thread.gy}`;
          const address = thread.valid ? `@${thread.address}` : "OOB";
          const label = thread.valid
            ? `block ${bx},${by}，thread ${thread.tx},${thread.ty}，全局坐标 ${thread.gx},${thread.gy}，地址 ${thread.address}`
            : `block ${bx},${by}，thread ${thread.tx},${thread.ty}，全局坐标 ${thread.gx},${thread.gy}，越界`;
          return `<button type="button" class="gp-thread-cell" data-block="${blockIndex}" data-lane="${thread.lane}" data-state="${thread.valid ? "valid" : "oob"}"${active ? " data-active" : ""} aria-label="${label}" title="${label}">
            <b>t${thread.tx}${config.vector ? "" : `,${thread.ty}`}</b><span>${coordinate}</span><em>${address}</em>
          </button>`;
        }).join("");
        blocks.push(`<article class="gp-block" data-block-index="${blockIndex}"${blockIndex === selectedBlock ? " data-selected" : ""}>
          <header><b>block (${bx}, ${by})</b><span>${config.blockX * config.blockY} threads</span></header>
          <div class="gp-block__threads" style="--gp-block-x:${config.blockX}">${cells}</div>
        </article>`);
      }
      get("[data-gp-grid]").innerHTML = blocks.join("");
    }

    function renderThreadTable() {
      const rows = selectedThreads().map((thread) => {
        const active = thread.lane === stepIndex;
        const threadCoord = config.vector ? `(${thread.tx})` : `(${thread.tx}, ${thread.ty})`;
        const globalCoord = config.vector ? `(${thread.gx})` : `(${thread.gx}, ${thread.gy})`;
        return `<tr data-state="${thread.valid ? "valid" : "oob"}"${active ? " data-active" : ""}>
          <td>${thread.lane}<small>W${thread.warp}</small></td>
          <td>${threadCoord}</td>
          <td>${globalCoord}</td>
          <td>${thread.valid ? thread.address : "guard"}</td>
          <td>${thread.valid ? `<span data-segment-color="${segmentClass(thread.segment)}">S${thread.segment}</span>` : "—"}</td>
        </tr>`;
      }).join("");
      get("[data-gp-thread-table]").innerHTML = rows;
    }

    function accessComparison(mode) {
      const all = selectedThreads(mode);
      const activeWarp = Math.floor(Math.max(0, stepIndex) / 32);
      const warpThreads = all.filter((thread) => thread.warp === activeWarp);
      const valid = warpThreads.filter((thread) => thread.valid);
      const segments = [...new Set(valid.map((thread) => thread.segment))];
      const addresses = warpThreads.map((thread) => {
        if (!thread.valid) return `<span data-oob><b>L${thread.lane}</b><em>OOB</em></span>`;
        return `<span data-segment-color="${segmentClass(thread.segment)}"><b>L${thread.lane}</b><em>@${thread.address}</em><small>S${thread.segment}</small></span>`;
      }).join("");
      return `<div class="gp-access-row"${config.mode === mode ? " data-current" : ""}>
        <header><b>${mode}</b><span>${mode === "contiguous" ? "addr = linear" : `addr = linear × ${config.stride}`}</span><strong>${segments.length} 个 32B 组</strong></header>
        <div>${addresses}</div>
        <p>warp ${activeWarp}：${valid.length} 个有效 lane，覆盖 ${segments.length} 个地址组${valid.length ? `（S${Math.min(...segments)}–S${Math.max(...segments)}）` : ""}</p>
      </div>`;
    }

    function renderComparison() {
      get("[data-gp-compare]").innerHTML =
        accessComparison("contiguous") + accessComparison("strided");
    }

    function currentDescription() {
      const thread = selectedThreads()[stepIndex];
      const { bx, by } = blockCoordinates(selectedBlock);
      if (!thread) return "没有当前线程";
      if (!thread.valid) {
        return `block (${bx}, ${by}) lane ${thread.lane} → global (${thread.gx}, ${thread.gy})，越界，由 guard 跳过。`;
      }
      return `block (${bx}, ${by}) lane ${thread.lane} → global (${thread.gx}, ${thread.gy}) → 地址 ${thread.address}，warp ${thread.warp}，32B 组 S${thread.segment}。`;
    }

    function renderStatus() {
      const description = currentDescription();
      get("[data-gp-status]").textContent = description;
    }

    function render() {
      renderSummary();
      renderGrid();
      renderThreadTable();
      renderComparison();
      renderStatus();
    }

    function announce(message = currentDescription()) {
      get("[data-gp-live]").textContent = message;
    }

    function stopPlaying() {
      if (playTimer) window.clearInterval(playTimer);
      playTimer = null;
      get("[data-gp-play]").textContent = "播放 block";
    }

    function moveStep(delta, shouldAnnounce = true) {
      const count = config.blockX * config.blockY;
      stepIndex = Math.min(count - 1, Math.max(0, stepIndex + delta));
      if (stepIndex === count - 1) stopPlaying();
      render();
      if (shouldAnnounce) announce();
    }

    function togglePlay() {
      if (playTimer) {
        stopPlaying();
        announce(`已暂停。${currentDescription()}`);
        return;
      }
      const count = config.blockX * config.blockY;
      if (stepIndex >= count - 1) stepIndex = 0;
      get("[data-gp-play]").textContent = "暂停";
      render();
      announce(`开始播放选定 block。${currentDescription()}`);
      playTimer = window.setInterval(() => moveStep(1, true), reducedMotion ? 1100 : 700);
    }

    function applyShapeChange() {
      const vector = controls.shape.value === "vector";
      if (vector) {
        matrixHeight = clamp(controls.height.value, 1, 12);
        matrixBlockY = clamp(controls.blockY.value, 1, 8);
        controls.height.value = 1;
        controls.blockY.value = 1;
      } else {
        controls.height.value = matrixHeight;
        controls.blockY.value = matrixBlockY;
      }
      controls.height.disabled = vector;
      controls.blockY.disabled = vector;
      controls.heightLabel.hidden = vector;
    }

    function updateConfiguration(message = "参数已更新。") {
      stopPlaying();
      config = readConfig();
      selectedBlock = Math.min(selectedBlock, totalBlocks() - 1);
      stepIndex = Math.min(stepIndex, config.blockX * config.blockY - 1);
      syncBlockOptions();
      render();
      announce(`${message} gridDim ${config.gridX} × ${config.gridY}，blockDim ${config.blockX} × ${config.blockY}。`);
    }

    Object.entries(controls).forEach(([name, element]) => {
      if (!element || name === "heightLabel" || name === "block") return;
      element.addEventListener("change", () => {
        if (name === "shape") applyShapeChange();
        updateConfiguration();
      });
      if (element.tagName === "INPUT") {
        element.addEventListener("input", () => updateConfiguration());
      }
    });

    controls.block.addEventListener("change", () => {
      stopPlaying();
      selectedBlock = clamp(controls.block.value, 0, totalBlocks() - 1);
      stepIndex = 0;
      render();
      announce(`已选择 ${controls.block.selectedOptions[0].textContent}。${currentDescription()}`);
    });

    get("[data-gp-prev]").addEventListener("click", () => moveStep(-1));
    get("[data-gp-next]").addEventListener("click", () => moveStep(1));
    get("[data-gp-play]").addEventListener("click", togglePlay);
    get("[data-gp-reset]").addEventListener("click", () => {
      stopPlaying();
      stepIndex = 0;
      render();
      announce(`已回到选定 block 的 lane 0。${currentDescription()}`);
    });

    get("[data-gp-grid]").addEventListener("click", (event) => {
      const cell = event.target.closest("[data-block][data-lane]");
      if (!cell) return;
      stopPlaying();
      selectedBlock = Number(cell.dataset.block);
      stepIndex = Number(cell.dataset.lane);
      controls.block.value = String(selectedBlock);
      render();
      announce(`已从网格选择线程。${currentDescription()}`);
    });

    lab.addEventListener("keydown", (event) => {
      if (event.target.matches("input, select, button, textarea, a")) return;
      if (event.key === "ArrowRight") {
        event.preventDefault();
        moveStep(1);
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        moveStep(-1);
      } else if (event.key === " ") {
        event.preventDefault();
        togglePlay();
      }
    });

    document.addEventListener("visibilitychange", () => {
      if (document.hidden) stopPlaying();
    });

    applyShapeChange();
    config = readConfig();
    syncBlockOptions();
    render();
    announce(`沙盘已初始化。${currentDescription()}`);
  }
})();
