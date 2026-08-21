(() => {
  const roots = document.querySelectorAll('[data-guide="minbpe"]');
  roots.forEach((root) => initLab(root));

  function initLab(root) {
    const lab = root.querySelector("[data-mb-lab]");
    if (!lab) return;

    const query = (selector) => lab.querySelector(selector);
    const encoder = new TextEncoder();
    const decoder = new TextDecoder("utf-8");
    const fatalDecoder = new TextDecoder("utf-8", { fatal: true });
    const specialText = "<|endoftext|>";
    const specialId = 100000;
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let state;
    let playTimer = null;

    function createBaseVocab() {
      return new Map(Array.from({ length: 256 }, (_, id) => [id, Uint8Array.of(id)]));
    }

    function clampLimit() {
      const input = query("[data-mb-limit]");
      const raw = Number(input.value);
      const value = Number.isFinite(raw) ? Math.max(1, Math.min(24, Math.floor(raw))) : 8;
      input.value = String(value);
      return value;
    }

    function ordinaryChunks(text, mode) {
      if (!text) return [];
      if (mode === "basic") return [text];
      return text.match(/ ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+/gu) || [];
    }

    function splitInput(text, mode, protectSpecial) {
      const parts = [];
      if (!protectSpecial) {
        ordinaryChunks(text, mode).forEach((part) => parts.push({ type: "ordinary", text: part }));
      } else {
        let offset = 0;
        while (offset < text.length) {
          const index = text.indexOf(specialText, offset);
          if (index === -1) {
            ordinaryChunks(text.slice(offset), mode).forEach((part) => parts.push({ type: "ordinary", text: part }));
            break;
          }
          ordinaryChunks(text.slice(offset, index), mode).forEach((part) => parts.push({ type: "ordinary", text: part }));
          parts.push({ type: "special", text: specialText });
          offset = index + specialText.length;
        }
        if (text === "") return [];
      }
      return parts.map((part) => ({
        ...part,
        ids: part.type === "special" ? [specialId] : [...encoder.encode(part.text)],
      }));
    }

    function reset(announceReset = true) {
      stopPlaying();
      const text = query("[data-mb-training]").value;
      const mode = query("[data-mb-mode]").value;
      const protectSpecial = query("[data-mb-special]").checked;
      state = {
        text,
        mode,
        protectSpecial,
        limit: clampLimit(),
        round: 0,
        chunks: splitInput(text, mode, protectSpecial),
        merges: [],
        vocab: createBaseVocab(),
        lastRule: null,
        stopped: false,
      };
      if (protectSpecial) state.vocab.set(specialId, encoder.encode(specialText));
      render();
      runCodec(false);
      if (announceReset) {
        announce(`训练已重置：${byteCount()} 个 UTF-8 bytes，${state.chunks.length} 个 chunk。`);
      }
    }

    function pairKey(left, right) {
      return `${left},${right}`;
    }

    function pairStats(chunks = state.chunks) {
      const counts = new Map();
      chunks.filter((chunk) => chunk.type === "ordinary").forEach((chunk) => {
        for (let index = 0; index < chunk.ids.length - 1; index += 1) {
          const key = pairKey(chunk.ids[index], chunk.ids[index + 1]);
          counts.set(key, (counts.get(key) || 0) + 1);
        }
      });
      return counts;
    }

    function parsePair(key) {
      return key.split(",").map(Number);
    }

    function bestPair(stats) {
      let selected = null;
      let bestCount = -1;
      stats.forEach((count, key) => {
        if (count > bestCount) {
          selected = key;
          bestCount = count;
        }
      });
      return selected ? { key: selected, count: bestCount, pair: parsePair(selected) } : null;
    }

    function mergeIds(ids, left, right, newId) {
      const merged = [];
      let index = 0;
      while (index < ids.length) {
        if (index < ids.length - 1 && ids[index] === left && ids[index + 1] === right) {
          merged.push(newId);
          index += 2;
        } else {
          merged.push(ids[index]);
          index += 1;
        }
      }
      return merged;
    }

    function concatBytes(left, right) {
      const output = new Uint8Array(left.length + right.length);
      output.set(left, 0);
      output.set(right, left.length);
      return output;
    }

    function step() {
      if (state.round >= state.limit || state.stopped) {
        stopPlaying();
        announce(state.stopped ? "没有可合并的普通 token pair。" : `已达到 ${state.limit} 次 merge 上限。`);
        return;
      }
      const stats = pairStats();
      const selected = bestPair(stats);
      if (!selected) {
        state.stopped = true;
        stopPlaying();
        render();
        announce("没有可合并的普通 token pair。 ");
        return;
      }

      const [left, right] = selected.pair;
      const id = 256 + state.round;
      state.chunks.forEach((chunk) => {
        if (chunk.type === "ordinary") chunk.ids = mergeIds(chunk.ids, left, right, id);
      });
      state.vocab.set(id, concatBytes(state.vocab.get(left), state.vocab.get(right)));
      state.lastRule = { id, left, right, count: selected.count };
      state.merges.push(state.lastRule);
      state.round += 1;
      render();
      runCodec(false);
      announce(`第 ${state.round} 轮：pair (${left}, ${right}) 出现 ${selected.count} 次，合并为 token ${id}。`);
      if (state.round >= state.limit) stopPlaying();
    }

    function togglePlay() {
      if (playTimer) {
        stopPlaying();
        announce("已暂停训练。 ");
        return;
      }
      if (state.round >= state.limit || state.stopped) reset(false);
      query("[data-mb-play]").textContent = "暂停";
      step();
      if (state.round < state.limit && !state.stopped) {
        playTimer = window.setInterval(step, reducedMotion ? 1000 : 700);
      }
    }

    function stopPlaying() {
      if (playTimer) window.clearInterval(playTimer);
      playTimer = null;
      const button = query("[data-mb-play]");
      if (button) button.textContent = "连续训练";
    }

    function encodeWithCurrentMerges(text) {
      const chunks = splitInput(text, state.mode, state.protectSpecial);
      let changed = true;
      while (changed) {
        changed = false;
        const stats = pairStats(chunks);
        const rule = state.merges.find((candidate) => stats.has(pairKey(candidate.left, candidate.right)));
        if (!rule) break;
        chunks.forEach((chunk) => {
          if (chunk.type === "ordinary") chunk.ids = mergeIds(chunk.ids, rule.left, rule.right, rule.id);
        });
        changed = true;
      }
      return chunks;
    }

    function decodeIds(ids) {
      const byteArrays = ids.map((id) => state.vocab.get(id)).filter(Boolean);
      const total = byteArrays.reduce((sum, bytes) => sum + bytes.length, 0);
      const output = new Uint8Array(total);
      let offset = 0;
      byteArrays.forEach((bytes) => {
        output.set(bytes, offset);
        offset += bytes.length;
      });
      return decoder.decode(output);
    }

    function runCodec(announceResult = true) {
      const text = query("[data-mb-encode-text]").value;
      const chunks = encodeWithCurrentMerges(text);
      const ids = chunks.flatMap((chunk) => chunk.ids);
      query("[data-mb-encoded]").innerHTML = renderChunks(chunks);
      query("[data-mb-ids]").textContent = ids.length ? `[${ids.join(", ")}]` : "[]";
      query("[data-mb-decoded]").textContent = decodeIds(ids);
      if (announceResult) announce(`使用 ${state.merges.length} 条 merge rules 编码为 ${ids.length} 个 token，并解码回输入文本。`);
    }

    function render() {
      const stats = pairStats();
      const selected = bestPair(stats);
      query("[data-mb-round]").textContent = `${state.round} / ${state.limit}`;
      query("[data-mb-token-count]").textContent = String(tokenCount());
      query("[data-mb-vocab-size]").textContent = String(256 + state.merges.length);
      query("[data-mb-choice]").textContent = state.lastRule
        ? `(${state.lastRule.left}, ${state.lastRule.right}) → ${state.lastRule.id}`
        : selected
          ? `候选 (${selected.pair.join(", ")})`
          : "—";
      query("[data-mb-sequence]").innerHTML = renderChunks(state.chunks);
      query("[data-mb-bytes]").textContent = [...encoder.encode(state.text)].map(hexByte).join(" ") || "—";
      query("[data-mb-step]").disabled = state.round >= state.limit || !selected;
      renderPairs(stats);
      renderRules();
      renderVocab();
    }

    function renderChunks(chunks) {
      if (!chunks.length) return '<span class="mb-empty">空序列</span>';
      return chunks.map((chunk, chunkIndex) => chunk.ids.map((id, tokenIndex) => {
        const classes = ["mb-token"];
        if (chunkIndex > 0 && tokenIndex === 0) classes.push("is-boundary");
        if (chunk.type === "special") classes.push("is-special");
        if (state.lastRule && id === state.lastRule.id) classes.push("is-new");
        const bytes = state.vocab.get(id) || new Uint8Array();
        return `<span class="${classes.join(" ")}" title="bytes: ${bytesHex(bytes)}"><b>${id}</b><small>${escapeHtml(tokenText(bytes, id))}</small></span>`;
      }).join("")).join("");
    }

    function renderPairs(stats) {
      const entries = [...stats.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10);
      const topKey = entries[0]?.[0];
      query("[data-mb-pairs]").innerHTML = entries.length ? entries.map(([key, count]) => {
        const [left, right] = parsePair(key);
        return `<div class="mb-pair-row"${key === topKey ? ' data-top="true"' : ""}><span>${tokenCompact(left)} + ${tokenCompact(right)}</span><b>${count}</b><i style="--mb-count:${Math.min(100, count * 14)}%"></i></div>`;
      }).join("") : '<p class="mb-empty">没有相邻 pair</p>';
    }

    function renderRules() {
      query("[data-mb-rules]").innerHTML = state.merges.length ? state.merges.map((rule, index) =>
        `<li${index === state.merges.length - 1 ? ' data-latest="true"' : ""}><span>#${index + 1}</span><code>(${rule.left}, ${rule.right}) → ${rule.id}</code><small>count ${rule.count}</small></li>`,
      ).join("") : '<li class="mb-empty">尚未训练 merge rule</li>';
    }

    function renderVocab() {
      query("[data-mb-vocab]").innerHTML = state.merges.length ? state.merges.map((rule) => {
        const bytes = state.vocab.get(rule.id);
        return `<span class="mb-vocab-item"><b>${rule.id}</b><i>${escapeHtml(tokenText(bytes, rule.id))}</i><code>${bytesHex(bytes)}</code></span>`;
      }).join("") : '<p class="mb-empty">vocab = 256 个 byte tokens</p>';
    }

    function tokenCompact(id) {
      const bytes = state.vocab.get(id);
      return `${id}·${escapeHtml(tokenText(bytes, id))}`;
    }

    function tokenText(bytes, id) {
      if (id === specialId) return specialText;
      if (!bytes?.length) return "?";
      try {
        const text = fatalDecoder.decode(bytes);
        if (text === " ") return "␠";
        if (text === "\n") return "↵";
        if (/^[\p{C}\s]+$/u.test(text)) return bytesHex(bytes);
        return text;
      } catch {
        return bytesHex(bytes);
      }
    }

    function bytesHex(bytes) {
      return [...bytes].map(hexByte).join(" ");
    }

    function hexByte(byte) {
      return byte.toString(16).toUpperCase().padStart(2, "0");
    }

    function tokenCount() {
      return state.chunks.reduce((sum, chunk) => sum + chunk.ids.length, 0);
    }

    function byteCount() {
      return encoder.encode(state.text).length;
    }

    function escapeHtml(value) {
      return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
    }

    function announce(message) {
      query("[data-mb-live]").textContent = message;
    }

    query("[data-mb-reset]").addEventListener("click", () => reset());
    query("[data-mb-step]").addEventListener("click", step);
    query("[data-mb-play]").addEventListener("click", togglePlay);
    query("[data-mb-encode]").addEventListener("click", () => runCodec());
    query("[data-mb-mode]").addEventListener("change", () => reset());
    query("[data-mb-special]").addEventListener("change", () => reset());
    query("[data-mb-limit]").addEventListener("change", () => reset());
    query("[data-mb-training]").addEventListener("change", () => reset());
    query("[data-mb-encode-text]").addEventListener("keydown", (event) => {
      if (event.key === "Enter") runCodec();
    });
    lab.addEventListener("keydown", (event) => {
      if (["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(event.target.tagName)) return;
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
