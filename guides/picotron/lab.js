(() => {
  const guideRoots = document.querySelectorAll('[data-guide="picotron"]');

  const integerValue = (input, fallback = 1) => {
    const value = Number(input?.value);
    return Number.isInteger(value) && value > 0 ? value : fallback;
  };

  const rankAt = (d, p, c, t, sizes) =>
    (((d * sizes.pp + p) * sizes.cp + c) * sizes.tp) + t;

  const coordinatesOf = (rank, sizes) => {
    let rest = rank;
    const t = rest % sizes.tp;
    rest = Math.floor(rest / sizes.tp);
    const c = rest % sizes.cp;
    rest = Math.floor(rest / sizes.cp);
    const p = rest % sizes.pp;
    const d = Math.floor(rest / sizes.pp);
    return { d, p, c, t };
  };

  const groupRanks = (axis, point, sizes) => {
    const count = sizes[axis];
    return Array.from({ length: count }, (_, index) => rankAt(
      axis === "dp" ? index : point.d,
      axis === "pp" ? index : point.p,
      axis === "cp" ? index : point.c,
      axis === "tp" ? index : point.t,
      sizes,
    ));
  };

  const makeSchedule = (stages, microBatches) => {
    const tasks = new Map();

    for (let stage = 0; stage < stages; stage += 1) {
      const warmup = Math.min(stages - stage - 1, microBatches);
      const sequence = [];

      for (let micro = 0; micro < warmup; micro += 1) {
        sequence.push({ kind: "F", stage, micro });
      }
      for (let offset = 0; offset < microBatches - warmup; offset += 1) {
        sequence.push({ kind: "F", stage, micro: warmup + offset });
        sequence.push({ kind: "B", stage, micro: offset });
      }
      for (let micro = microBatches - warmup; micro < microBatches; micro += 1) {
        sequence.push({ kind: "B", stage, micro });
      }

      sequence.forEach((task, index) => {
        const key = `${task.kind}-${task.stage}-${task.micro}`;
        tasks.set(key, {
          ...task,
          key,
          previous: index > 0
            ? `${sequence[index - 1].kind}-${stage}-${sequence[index - 1].micro}`
            : null,
        });
      });
    }

    const memo = new Map();
    const visiting = new Set();
    const finish = (key) => {
      if (memo.has(key)) return memo.get(key);
      if (visiting.has(key)) return 0;
      visiting.add(key);

      const task = tasks.get(key);
      const dependencies = [];
      if (task.previous) dependencies.push(task.previous);
      if (task.kind === "F" && task.stage > 0) {
        dependencies.push(`F-${task.stage - 1}-${task.micro}`);
      }
      if (task.kind === "B") {
        dependencies.push(`F-${task.stage}-${task.micro}`);
        if (task.stage < stages - 1) {
          dependencies.push(`B-${task.stage + 1}-${task.micro}`);
        }
      }

      const start = dependencies.reduce(
        (latest, dependency) => Math.max(latest, finish(dependency)),
        0,
      );
      const end = start + 1;
      task.start = start;
      memo.set(key, end);
      visiting.delete(key);
      return end;
    };

    tasks.forEach((task) => finish(task.key));
    const span = Math.max(...Array.from(tasks.values(), (task) => task.start + 1));
    return { tasks: Array.from(tasks.values()), span };
  };

  guideRoots.forEach((guide) => {
    const lab = guide.querySelector("[data-picotron-lab]");
    if (!lab) return;

    const inputs = Object.fromEntries(
      Array.from(lab.querySelectorAll("[data-picotron-input]"), (input) => [
        input.dataset.picotronInput,
        input,
      ]),
    );
    const status = lab.querySelector("[data-picotron-status]");
    const live = lab.querySelector("[data-picotron-live]");
    const rankSelect = lab.querySelector("[data-picotron-rank]");
    const topology = lab.querySelector("[data-picotron-topology]");
    const groups = lab.querySelector("[data-picotron-groups]");
    const timeline = lab.querySelector("[data-picotron-timeline]");
    const bubble = lab.querySelector("[data-picotron-bubble]");
    const shapes = lab.querySelector("[data-picotron-shapes]");

    let selectedRank = 0;
    let current = null;

    const readConfiguration = () => ({
      gpus: integerValue(inputs.gpus),
      dp: integerValue(inputs.dp),
      tp: integerValue(inputs.tp),
      pp: integerValue(inputs.pp),
      cp: integerValue(inputs.cp),
      micro: integerValue(inputs.micro),
      sequence: integerValue(inputs.sequence),
      hidden: integerValue(inputs.hidden),
    });

    const renderStatus = (config) => {
      const product = config.dp * config.tp * config.pp * config.cp;
      const topologyValid = product === config.gpus && config.gpus <= 64;
      const sequenceValid = config.sequence % config.cp === 0;
      const hiddenValid = config.hidden % config.tp === 0;
      const valid = topologyValid && sequenceValid && hiddenValid;
      const issues = [];

      if (!topologyValid) issues.push(`并行乘积是 ${product}，需要等于 GPU 总数 ${config.gpus}`);
      if (!sequenceValid) issues.push(`序列长度 ${config.sequence} 不能被 CP=${config.cp} 整除`);
      if (!hiddenValid) issues.push(`隐藏维度 ${config.hidden} 不能被 TP=${config.tp} 整除`);

      status.className = `picotron-status ${valid ? "is-valid" : "is-invalid"}`;
      status.innerHTML = `
        <div><span>${valid ? "配置通过" : "需要调整"}</span><b>${config.dp} DP × ${config.tp} TP × ${config.pp} PP × ${config.cp} CP = ${product}</b></div>
        <p>${issues.length ? issues.join("；") : `使用 ${config.gpus} 个 rank，序列与隐藏维度可整除。`}</p>
      `;
      return { valid, topologyValid, sequenceValid, hiddenValid, product, issues };
    };

    const renderRankSelect = (config) => {
      selectedRank = Math.min(selectedRank, config.gpus - 1);
      rankSelect.innerHTML = Array.from({ length: config.gpus }, (_, rank) =>
        `<option value="${rank}"${rank === selectedRank ? " selected" : ""}>rank ${rank}</option>`,
      ).join("");
    };

    const renderGroups = (config) => {
      const point = coordinatesOf(selectedRank, config);
      const definitions = [
        ["TP", "tp", "同层张量"],
        ["CP", "cp", "序列分片"],
        ["PP", "pp", "相邻 stages"],
        ["DP", "dp", "数据副本"],
      ];

      groups.innerHTML = `
        <div class="picotron-coordinate">
          <span>rank ${selectedRank}</span>
          <code>[d${point.d}, p${point.p}, c${point.c}, t${point.t}]</code>
        </div>
        ${definitions.map(([label, axis, detail]) => `
          <div class="picotron-group">
            <div><b>${label} group</b><small>${detail}</small></div>
            <p>${groupRanks(axis, point, config).map((rank) =>
              `<span${rank === selectedRank ? ' class="is-selected"' : ""}>${rank}</span>`,
            ).join("")}</p>
          </div>
        `).join("")}
      `;
    };

    const renderTopology = (config, topologyValid) => {
      if (!topologyValid) {
        topology.innerHTML = '<p class="picotron-empty">乘积匹配后显示 rank topology。</p>';
        groups.innerHTML = "";
        rankSelect.innerHTML = "";
        rankSelect.disabled = true;
        return;
      }

      rankSelect.disabled = false;
      renderRankSelect(config);
      topology.innerHTML = Array.from({ length: config.dp }, (_, d) => `
        <section class="picotron-replica">
          <h4><span>DP ${d}</span><small>model replica</small></h4>
          <div class="picotron-stages">
            ${Array.from({ length: config.pp }, (_, p) => `
              <div class="picotron-stage">
                <h5>PP ${p}</h5>
                <div class="picotron-ranks" style="--picotron-tp:${Math.min(config.tp, 8)}">
                  ${Array.from({ length: config.cp }, (_, c) =>
                    Array.from({ length: config.tp }, (_, t) => {
                      const rank = rankAt(d, p, c, t, config);
                      return `<button type="button" data-picotron-rank-button="${rank}" class="${rank === selectedRank ? "is-selected" : ""}" aria-label="rank ${rank}，DP ${d}，PP ${p}，CP ${c}，TP ${t}" title="d${d} / p${p} / c${c} / t${t}"><b>${rank}</b><small>c${c}·t${t}</small></button>`;
                    }).join(""),
                  ).join("")}
                </div>
              </div>
            `).join("")}
          </div>
        </section>
      `).join("");
      renderGroups(config);
    };

    const renderTimeline = (config) => {
      const stages = Math.min(config.pp, 16);
      const microBatches = Math.min(config.micro, 32);
      const schedule = makeSchedule(stages, microBatches);
      const byStageAndTime = new Map(
        schedule.tasks.map((task) => [`${task.stage}-${task.start}`, task]),
      );
      const bubbleRatio = (stages - 1) / (microBatches + stages - 1);

      bubble.innerHTML = `<b>${(bubbleRatio * 100).toFixed(1)}%</b><span>理论 bubble</span>`;
      timeline.style.setProperty("--picotron-slots", schedule.span);
      timeline.innerHTML = `
        <div class="picotron-time-label">stage</div>
        <div class="picotron-time-axis">${Array.from({ length: schedule.span }, (_, time) => `<span>${time}</span>`).join("")}</div>
        ${Array.from({ length: stages }, (_, stage) => `
          <div class="picotron-stage-label">P${stage}</div>
          <div class="picotron-stage-track">
            ${Array.from({ length: schedule.span }, (_, time) => {
              const task = byStageAndTime.get(`${stage}-${time}`);
              if (!task) return '<span class="is-idle" title="bubble" aria-label="空闲"></span>';
              const word = task.kind === "F" ? "forward" : "backward";
              return `<span class="is-${word}" title="${word} micro-batch ${task.micro + 1}" aria-label="${word} micro-batch ${task.micro + 1}">${task.kind}${task.micro + 1}</span>`;
            }).join("")}
          </div>
        `).join("")}
      `;
    };

    const renderShapes = (config, validity) => {
      if (!validity.sequenceValid || !validity.hiddenValid) {
        shapes.innerHTML = '<p class="picotron-empty">序列长度需被 CP 整除，隐藏维度需被 TP 整除。</p>';
        return;
      }

      const localSequence = config.sequence / config.cp;
      const localHidden = config.hidden / config.tp;
      shapes.innerHTML = `
        <div class="picotron-shape-chain">
          <div><span>global activation</span><code>[Bμ, ${config.sequence}, ${config.hidden}]</code><small>完整序列与隐藏维</small></div>
          <i aria-hidden="true">CP ÷ ${config.cp} →</i>
          <div><span>CP-local</span><code>[Bμ, ${localSequence}, ${config.hidden}]</code><small>本 rank 的 token 区间</small></div>
          <i aria-hidden="true">TP ÷ ${config.tp} →</i>
          <div><span>TP × CP local</span><code>[Bμ, ${localSequence}, ${localHidden}]</code><small>column projection 内部</small></div>
        </div>
        <div class="picotron-shape-notes">
          <p><b>PP stage 边界</b><code>[Bμ, ${localSequence}, ${config.hidden}]</code><span>row projection all-reduce 后恢复 hidden 维，再传给下一 stage。</span></p>
          <p><b>Column weight / rank</b><code>[${localHidden}, ${config.hidden}]</code><span>输出特征按 TP 切分。</span></p>
          <p><b>CP attention</b><code>${config.cp - 1} ring hops</code><span>本地 Q 保留，K/V block 沿 CP group 移动。</span></p>
        </div>
      `;
    };

    const update = (announcement = true) => {
      const config = readConfiguration();
      current = config;
      const validity = renderStatus(config);
      renderTopology(config, validity.topologyValid);
      renderTimeline(config);
      renderShapes(config, validity);

      if (announcement) {
        live.textContent = validity.valid
          ? `配置通过。${config.gpus} 个 rank，理论 pipeline bubble ${(100 * (config.pp - 1) / (config.micro + config.pp - 1)).toFixed(1)}%。`
          : `配置需要调整：${validity.issues.join("；")}。`;
      }
    };

    lab.addEventListener("input", (event) => {
      if (event.target.matches("[data-picotron-input]")) update();
    });

    rankSelect.addEventListener("change", () => {
      selectedRank = Number(rankSelect.value);
      renderTopology(current, true);
      const point = coordinatesOf(selectedRank, current);
      live.textContent = `已选择 rank ${selectedRank}：DP ${point.d}，PP ${point.p}，CP ${point.c}，TP ${point.t}。`;
    });

    topology.addEventListener("click", (event) => {
      const button = event.target.closest("[data-picotron-rank-button]");
      if (!button) return;
      selectedRank = Number(button.dataset.picotronRankButton);
      renderTopology(current, true);
      const point = coordinatesOf(selectedRank, current);
      live.textContent = `已选择 rank ${selectedRank}：DP ${point.d}，PP ${point.p}，CP ${point.c}，TP ${point.t}。`;
    });

    update(false);
  });
})();
