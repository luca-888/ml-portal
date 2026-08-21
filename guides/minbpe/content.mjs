const upstream = "https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602";

export default {
  slug: "minbpe",
  title: "minbpe",
  subtitle: "从 UTF-8 bytes 训练 BPE 合并规则，并用固定规则完成编码与解码。",
  repoUrl: "https://github.com/karpathy/minbpe",
  accent: "#2563eb",
  tags: ["BPE", "Tokenizer", "UTF-8", "Regex", "Special Tokens"],
  audience: "想理解语言模型 tokenizer 的训练产物、编码顺序和字节边界的学习者。",
  prerequisites: [
    "知道字符串、字节序列和整数 token id 是不同表示",
    "了解字典计数和相邻元素遍历",
    "能阅读 Python 循环与正则表达式",
  ],
  outcomes: [
    "把 Unicode 文本展开为 UTF-8 byte ids，并说明多字节字符的边界",
    "逐轮计算 pair frequency、创建 merge rule 和扩展词表",
    "区分训练 merges 与使用已有 merges 进行 encode",
    "解释 regex chunk、special token 和 decode 各自改变什么",
  ],
  overview: [
    "minbpe 实现 byte-level BPE。基础词表含 256 个单字节 token；训练时反复合并语料中频率最高的相邻 token pair，并从 256 起为新 token 分配 id。merge table 和由它派生的 vocab 是训练产物。",
    "页面基于官方仓库提交 1acefe8。实验完整处理浏览器 UTF-8 bytes；regex-aware 模式用 Unicode 字母、数字、空白和符号类别生成互不跨越的 chunk，用于展示 RegexTokenizer 的边界机制，不复刻其完整 GPT-4 正则。",
  ],
  architecture: {
    title: "训练路径与使用路径",
    description: "训练从语料统计新规则；使用阶段固定 merge table，只按规则优先级编码新文本。",
    nodes: [
      { id: "text", label: "Unicode text", detail: "Python 或浏览器中的字符串。" },
      { id: "split", label: "Regex / special split", detail: "可选地先切成互不合并的普通 chunk 与 special token。" },
      { id: "bytes", label: "UTF-8 bytes", detail: "每个普通 chunk 编码为 0 到 255 的整数。" },
      { id: "stats", label: "Pair counts", detail: "训练时统计每个 chunk 内相邻 token pair。" },
      { id: "merges", label: "Merge table", detail: "记录 pair 到新 token id；id 同时表示学习顺序。" },
      { id: "vocab", label: "Vocabulary", detail: "新 token 的 bytes 是两个子 token bytes 的拼接。" },
      { id: "encode", label: "Encode", detail: "对新文本反复应用当前可用且最早学习的 merge。" },
      { id: "decode", label: "Decode", detail: "按 id 查 bytes，拼接后用 UTF-8 解码。" },
    ],
    edges: [
      { from: "text", to: "split", label: "预处理" },
      { from: "split", to: "bytes", label: "encode UTF-8" },
      { from: "bytes", to: "stats", label: "训练" },
      { from: "stats", to: "merges", label: "最高频 pair" },
      { from: "merges", to: "vocab", label: "拼接 bytes" },
      { from: "bytes", to: "encode", label: "使用" },
      { from: "merges", to: "encode", label: "固定优先级" },
      { from: "vocab", to: "decode", label: "id → bytes" },
      { from: "encode", to: "decode", label: "token ids" },
    ],
  },
  sections: [
    {
      id: "utf8-bytes",
      title: "1. BPE 的叶子是 bytes，不是字符",
      summary: "文本先编码为 UTF-8，再把每个 byte 作为 0–255 的初始 token。",
      paragraphs: [
        "ASCII 字符通常占一个 byte，例如 A 是 65。中文“中”的 UTF-8 是三个 bytes：228、184、173。训练开始时它们是三个独立 token；后续 merge 可以把其中相邻部分合成一个 token。",
        "byte-level 方案的基础词表覆盖所有可能输入 bytes，因此不需要 unknown character。单个中间 token 可能只含一个多字节字符的一部分，单独显示时不是合法 UTF-8；完整 token 序列拼回 bytes 后才能可靠解码。",
      ],
      bullets: [
        "字符串长度、Unicode code point 数和 UTF-8 byte 数可能不同。",
        "基础 vocab 固定为 id 0–255 对应单个 byte。",
        "实验对不可独立解码的 token 显示十六进制 bytes。",
      ],
      code: {
        title: "BasicTokenizer 的字节预处理",
        language: "python",
        snippet: `def train(self, text, vocab_size, verbose=False):
    assert vocab_size >= 256
    num_merges = vocab_size - 256

    # input text preprocessing
    text_bytes = text.encode("utf-8") # raw bytes
    ids = list(text_bytes) # list of integers in range 0..255

    # iteratively merge the most common pairs to create new tokens
    merges = {} # (int, int) -> int
    vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes`,
        sourceUrl: `${upstream}/minbpe/basic.py#L20-L30`,
        sourceLabel: "minbpe/basic.py",
        note: "官方源码原文；截取训练初始化。",
      },
      visual: {
        title: "同一文本的字符与 byte 视图",
        html: `<div class="mb-byte-lanes" role="img" aria-label="A中在字符层有两个字符，在 UTF-8 层有四个 bytes">
          <div><b>字符</b><span>A</span><span>中</span></div>
          <div><b>UTF-8</b><span>41</span><span>E4</span><span>B8</span><span>AD</span></div>
          <div><b>初始 id</b><span>65</span><span>228</span><span>184</span><span>173</span></div>
        </div>`,
        caption: "十六进制 E4 B8 AD 合在一起才是“中”。",
      },
    },
    {
      id: "pair-merge",
      title: "2. 每轮合并最高频相邻 pair",
      summary: "统计当前 token 序列的相邻 pair，替换最高频项，并把新 token 加入规则表和词表。",
      paragraphs: [
        "get_stats 对 zip(ids, ids[1:]) 计数。选择频率最高的 pair 后，merge 从左到右把每个不重叠出现替换为新 id。例如 [1,2,1,2] 的 (1,2) 合并为 256 后得到 [256,256]。",
        "新 token 的 bytes 等于两个子 token bytes 的拼接。训练一轮会同时增加一条 merges[(p0,p1)] = idx 和一个 vocab[idx]；词表大小因此从 256 每轮增加 1。",
      ],
      bullets: [
        "频率在每轮替换后的新序列上重新统计。",
        "regex 模式在所有普通 chunk 内汇总计数，但不统计跨 chunk pair。",
        "相同最高频计数的选择依赖 pair 首次进入统计字典的顺序。",
      ],
      code: {
        title: "从左到右替换一个 pair",
        language: "python",
        snippet: `def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids`,
        sourceUrl: `${upstream}/minbpe/base.py#L25-L41`,
        sourceLabel: "minbpe/base.py",
        note: "官方源码原文；省略 docstring。",
      },
      visual: {
        title: "一次 merge 的状态变化",
        html: `<div class="mb-merge-flow" role="img" aria-label="pair 97 98 被替换为新 token 256">
          <span>97</span><span>98</span><span>97</span><span>98</span><i>→</i><span class="is-new">256</span><span class="is-new">256</span>
        </div>`,
        caption: "规则 (97, 98) → 256 同时定义 256 的 bytes 为 b'ab'。",
      },
    },
    {
      id: "train-versus-encode",
      title: "3. 训练 tokenizer 与使用 tokenizer 是两种算法",
      summary: "训练按频率发明规则；encode 不再看频率，只按已有 merge id 的先后顺序应用规则。",
      paragraphs: [
        "训练输入是语料和目标 vocab_size，输出是 merges 与 vocab。encode 输入是新文本和已经固定的 merges，输出 token ids；它不会因待编码文本中某个 pair 很常见而创建新 token。",
        "encode 每轮列出当前序列存在的 pairs，从其中选择 merge id 最小的规则。较小 id 表示训练时更早创建，必须先应用，否则后续规则依赖的子 token 可能还不存在。",
      ],
      bullets: [
        "train：最高 frequency 优先，修改 merge table。",
        "encode：最低 merge id 优先，不修改 merge table。",
        "同一 merge table 对相同 bytes 给出确定的 token ids。",
      ],
      code: {
        title: "BasicTokenizer 使用已训练规则编码",
        language: "python",
        snippet: `def encode(self, text):
    # given a string text, return the token ids
    text_bytes = text.encode("utf-8") # raw bytes
    ids = list(text_bytes) # list of integers in range 0..255
    while len(ids) >= 2:
        # find the pair with the lowest merge index
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
        if pair not in self.merges:
            break # nothing else can be merged anymore
        # otherwise let's merge the best pair (lowest merge index)
        idx = self.merges[pair]
        ids = merge(ids, pair, idx)
    return ids`,
        sourceUrl: `${upstream}/minbpe/basic.py#L57-L74`,
        sourceLabel: "minbpe/basic.py",
        note: "官方源码原文；省略两行终止条件注释，行为不变。",
      },
      visual: {
        title: "两条路径使用不同选择标准",
        html: `<div class="mb-path-compare" role="img" aria-label="训练按最高频率，编码按最低规则编号">
          <div><b>训练</b><span>当前语料</span><i>最高 pair count</i><span>新增 rule 256, 257…</span></div>
          <div><b>编码</b><span>新文本</span><i>最低已有 rule id</i><span>固定 token ids</span></div>
        </div>`,
        caption: "实验上半区训练规则，下半区用当前规则重新编码另一段文本。",
      },
    },
    {
      id: "regex-special",
      title: "4. Regex chunk 和 special token 都建立不可跨越的边界",
      summary: "RegexTokenizer 先切文本，再在每个普通 chunk 内独立执行 BPE；special token 走单独分支。",
      paragraphs: [
        "RegexTokenizer 默认使用 GPT-4 split pattern，把字母、数字、符号与换行等类别切成 chunks。训练统计会跨 chunks 汇总相同 pair 的次数，但 merge 只在各 chunk 内替换，因此无法产生横跨类别边界的 token。",
        "special token 不是靠 BPE 学出的 bytes 组合。register_special_tokens 显式注册字符串到 id；encode 只有在 allowed_special 允许时才把精确匹配整体映射为该 id，否则按普通文本处理或报错。",
      ],
      bullets: [
        "regex 先于 UTF-8 byte BPE 执行。",
        "special token id 应与普通 vocab id 空间避免冲突。",
        "decode 分别查询普通 vocab 和 inverse_special_tokens，再拼接 bytes。",
      ],
      code: {
        title: "按 regex chunk 分别编码后拼接",
        language: "python",
        snippet: `def encode_ordinary(self, text):
    """Encoding that ignores any special tokens."""
    # split text into chunks of text by categories defined in regex pattern
    text_chunks = re.findall(self.compiled_pattern, text)
    # all chunks of text are encoded separately, then results are joined
    ids = []
    for chunk in text_chunks:
        chunk_bytes = chunk.encode("utf-8") # raw bytes
        chunk_ids = self._encode_chunk(chunk_bytes)
        ids.extend(chunk_ids)
    return ids`,
        sourceUrl: `${upstream}/minbpe/regex.py#L111-L121`,
        sourceLabel: "minbpe/regex.py",
        note: "官方源码原文。",
      },
      visual: {
        title: "边界改变允许统计的 pairs",
        html: `<div class="mb-chunks" role="img" aria-label="Hello 123 感叹号被切为三个 chunk，pair 不能跨边界">
          <span data-type="letters"> Hello</span><em>边界</em><span data-type="numbers"> 123</span><em>边界</em><span data-type="punct">!</span>
        </div>`,
        caption: "chunk 内仍从 UTF-8 bytes 开始训练和编码。",
      },
    },
    {
      id: "vocab-decode",
      title: "5. Decode 只拼 bytes，不反演训练过程",
      summary: "vocab 把每个 id 映射到 bytes；decode 查表拼接，再做一次 UTF-8 解码。",
      paragraphs: [
        "vocab 可以从 merges 确定性重建：0–255 是单 bytes，后续 token bytes 是其 pair 两侧 token bytes 的连接。解码不需要重新执行 merge，也不需要知道 token boundary 原来对应几个字符。",
        "如果 token ids 顺序构成合法 UTF-8，拼接后还原原文。若 ids 缺失或顺序被破坏，普通 BasicTokenizer 使用 errors='replace' 产生替换字符；RegexTokenizer 对未知 id 先抛出 ValueError。",
      ],
      bullets: [
        ".model 保存 pattern、special tokens 和 merge pairs，可用于 load。",
        ".vocab 是面向人的展示文件，对不完整 UTF-8 使用替换字符，因此不能用于无损 load。",
        "token boundary 与字符 boundary 不必重合。",
      ],
      visual: {
        title: "token ids 到文本",
        html: `<div class="mb-decode-flow" role="img" aria-label="token id 通过 vocab 恢复 bytes 再解码为文本">
          <span>ids<br><b>[256, 257]</b></span><i>vocab 查表</i><span>bytes<br><b>F0 9F · 98 80</b></span><i>UTF-8 decode</i><span>text<br><b>😀</b></span>
        </div>`,
        caption: "单个 token 的 bytes 可以不是完整字符，拼接后的整体仍可合法解码。",
      },
    },
  ],
  lab: {
    title: "BPE 合并与编码工作台",
    intro: "输入训练文本，选择 basic 或 regex-aware，逐步合并当前最高频 pair。工作台显示 UTF-8 bytes、chunk 边界、pair 计数、merge rules 和词表增长；下方可用当前规则编码另一段文本并立即解码。键盘：→ 下一轮，空格播放/暂停，R 重置。",
    html: `<div class="mb-lab" data-mb-lab tabindex="0">
      <section class="mb-lab__controls" aria-label="BPE 训练参数">
        <label>训练文本<textarea rows="3" data-mb-training>banana banana 中文中文</textarea></label>
        <div class="mb-lab__options">
          <label>模式<select data-mb-mode><option value="basic">byte / basic</option><option value="regex">regex-aware</option></select></label>
          <label class="mb-check"><input type="checkbox" data-mb-special>保护 &lt;|endoftext|&gt;</label>
          <label>最多 merges<input type="number" min="1" max="24" value="8" data-mb-limit></label>
        </div>
        <div class="mb-lab__buttons"><button type="button" data-mb-reset>重置训练</button><button type="button" data-mb-step>下一次合并</button><button type="button" data-mb-play>连续训练</button></div>
      </section>

      <section class="mb-lab__metrics" aria-label="训练状态">
        <div><span>训练轮次</span><strong data-mb-round>0 / 8</strong></div>
        <div><span>token 数</span><strong data-mb-token-count>0</strong></div>
        <div><span>vocab 大小</span><strong data-mb-vocab-size>256</strong></div>
        <div><span>本轮选择</span><strong data-mb-choice>—</strong></div>
      </section>

      <section class="mb-lab__sequence">
        <div class="mb-panel-title"><h4>训练 token 序列</h4><span>粗间隔表示 regex / special boundary</span></div>
        <div class="mb-token-lane" data-mb-sequence></div>
        <div class="mb-byte-source"><b>原始 UTF-8 bytes</b><code data-mb-bytes></code></div>
      </section>

      <div class="mb-lab__middle">
        <section class="mb-lab__pairs">
          <div class="mb-panel-title"><h4>当前 pair 计数</h4><span>按次数降序</span></div>
          <div data-mb-pairs></div>
        </section>
        <section class="mb-lab__rules">
          <div class="mb-panel-title"><h4>merge table</h4><span>训练顺序即规则优先级</span></div>
          <ol data-mb-rules></ol>
        </section>
        <section class="mb-lab__vocab">
          <div class="mb-panel-title"><h4>新增 vocab</h4><span>基础 256 项省略</span></div>
          <div data-mb-vocab></div>
        </section>
      </div>

      <section class="mb-lab__codec">
        <div class="mb-panel-title"><h4>用当前 merges 编码 / 解码</h4><span>encode 不会新增规则</span></div>
        <div class="mb-codec-input"><label>待编码文本<input type="text" value="banana 中文" data-mb-encode-text></label><button type="button" data-mb-encode>重新 encode / decode</button></div>
        <div class="mb-codec-grid">
          <div><b>token boundaries</b><div class="mb-token-lane" data-mb-encoded></div></div>
          <div><b>token ids</b><code data-mb-ids>—</code></div>
          <div><b>decode</b><output data-mb-decoded>—</output></div>
        </div>
      </section>
      <div class="mb-lab__live" data-mb-live aria-live="polite" aria-atomic="true"></div>
    </div>`,
  },
  takeaways: [
    "byte-level BPE 先把文本变成 UTF-8 bytes，因此基础 256 token 可表示任意输入。",
    "训练按当前语料 pair frequency 新增规则；encode 只按已有规则 id 顺序应用 merge。",
    "regex chunk 限制允许合并的边界，special token 则绕过普通 BPE 路径。",
    "decode 通过 vocab 恢复并拼接 bytes；token boundary 不等于字符 boundary。",
  ],
  sources: [
    { label: "minbpe 官方仓库", url: "https://github.com/karpathy/minbpe" },
    { label: "公共辅助函数与 Tokenizer 基类", url: `${upstream}/minbpe/base.py` },
    { label: "BasicTokenizer", url: `${upstream}/minbpe/basic.py` },
    { label: "RegexTokenizer 与 special tokens", url: `${upstream}/minbpe/regex.py` },
    { label: "官方练习说明", url: `${upstream}/exercise.md` },
  ],
};
