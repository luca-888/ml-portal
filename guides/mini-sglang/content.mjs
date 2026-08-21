const upstream = "https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2";

export default {
  slug: "mini-sglang",
  title: "Mini-SGLang",
  subtitle: "从请求入口到分页 KV Cache 的大语言模型服务实现。",
  repoUrl: "https://github.com/sgl-project/mini-sglang",
  accent: "#3c63dd",
  tags: ["LLM Serving", "Radix Cache", "Scheduler", "Tensor Parallel"],
  audience: "读过 Transformer 推理流程，准备理解在线推理系统请求路径的开发者。",
  prerequisites: [
    "理解 prefill 与逐 token decode 的区别",
    "了解 attention 中 K、V 会跨 decode step 复用",
    "能阅读 Python、PyTorch tensor shape 与异步 CUDA stream 代码",
  ],
  outcomes: [
    "沿 HTTP、tokenizer、scheduler、engine、detokenizer 追踪一次请求",
    "解释 Radix Cache 如何按 token 前缀复用 KV 页并执行叶节点淘汰",
    "说明 Chunked Prefill 和 Overlap Scheduling 分别处理什么等待时间",
    "识别 Tensor Parallel rank 与 decode attention 的本地数据边界",
  ],
  overview: [
    "Mini-SGLang 把 API server、tokenizer、detokenizer 和每个 Tensor Parallel rank 的 scheduler 分成独立进程。控制消息通过 ZMQ 传递；多 GPU rank 之间通过分布式通信保持相同的请求与采样状态。",
    "调度器不是只维护一个请求队列。它还持有请求表、分页 KV 空间、前缀缓存、prefill 队列和 decode 集合。请求命中前缀后只计算未缓存的 token；长 prefill 可按预算切片；GPU 执行当前 batch 时，CPU 可以处理上一批结果并准备后续工作。",
    "本页先沿数据路径定位代码，再把前缀树、chunk 预算和 CPU/GPU 时间线放到同一个实验中。实验是机制模型，不给出真实硬件延迟。",
  ],
  architecture: {
    title: "一次生成请求经过的进程与设备边界",
    description: "实线表示请求或 token 流；TP rank 0 将请求同步给其他 scheduler，各 rank 驱动本地 Engine。",
    nodes: [
      { id: "client", label: "HTTP client", detail: "发送 completion 或 chat completion 请求。" },
      { id: "api", label: "API Server", detail: "分配 uid，通过 ZMQ 转交 TokenizeMsg，并流式返回文本。" },
      { id: "tokenizer", label: "Tokenizer", detail: "将 prompt 编码为 input_ids，产生 UserMsg。" },
      { id: "scheduler", label: "Scheduler · rank 0", detail: "接收消息，查询前缀缓存，组织 prefill 或 decode batch。" },
      { id: "cache", label: "Radix + page table", detail: "前缀树保存 token 到 KV 页位置的映射；page table 记录请求使用的物理槽位。" },
      { id: "engine", label: "Engine · each TP rank", detail: "持有模型分片、KV pool、attention backend、sampler 和 CUDA Graph。" },
      { id: "attention", label: "Decode attention", detail: "写入新 K/V，并通过分页表读取历史 K/V。" },
      { id: "detokenizer", label: "Detokenizer", detail: "将 next token 增量解码为文本并返回 API Server。" },
    ],
    edges: [
      { from: "client", to: "api", label: "HTTP" },
      { from: "api", to: "tokenizer", label: "TokenizeMsg" },
      { from: "tokenizer", to: "scheduler", label: "UserMsg" },
      { from: "scheduler", to: "cache", label: "match / allocate" },
      { from: "scheduler", to: "engine", label: "Batch" },
      { from: "cache", to: "attention", label: "KV locations" },
      { from: "engine", to: "attention", label: "Q/K/V" },
      { from: "attention", to: "detokenizer", label: "sampled token" },
      { from: "detokenizer", to: "api", label: "incremental text" },
    ],
  },
  sections: [
    {
      id: "request-path",
      title: "01 · 从 HTTP 到 Scheduler",
      summary: "文本先在独立 tokenizer 进程中变为 input_ids，rank 0 scheduler 再把 UserMsg 纳入 prefill 队列。",
      paragraphs: [
        "API Server 的职责是协议适配、uid 和流式连接；模型调度发生在 scheduler 进程。Tokenizer worker 同时接收 TokenizeMsg、DetokenizeMsg 与 AbortMsg，并分别把编码后的 UserMsg送给 backend，或把增量文本送回 frontend。",
        "使用多个 GPU 时，每个 TP rank 都有一个 scheduler 和 Engine。rank 0 从 tokenizer 接收请求，然后在 SchedulerIOMixin 中向其他 rank 广播，使各 rank 对同一批请求执行相同的调度步骤。",
      ],
      bullets: [
        "请求对象到达 scheduler 时已包含 CPU 上的 input_ids 与 sampling_params。",
        "scheduler 对最大序列长度做检查后，将请求放入 PrefillManager.pending_list。",
        "生成 token 经 DetokenizeMsg 回到 detokenizer，再以 UserReply 返回 API Server。",
      ],
      code: {
        title: "Tokenizer worker 构造 backend 消息",
        language: "python",
        snippet: `tensors = tokenize_manager.tokenize(tokenize_msg)
batch_output = BatchBackendMsg(
    data=[
        UserMsg(
            uid=msg.uid,
            input_ids=t,
            sampling_params=msg.sampling_params,
        )
        for msg, t in zip(tokenize_msg, tensors, strict=True)
    ]
)`,
        sourceUrl: `${upstream}/python/minisgl/tokenizer/server.py#L87-L98`,
        sourceLabel: "python/minisgl/tokenizer/server.py",
        note: "官方源码节选；省略单条消息解包与发送语句。",
      },
      visual: {
        title: "请求状态的所有权变化",
        html: `<div class="msg-route" aria-label="请求消息路径">
          <div><b>HTTP JSON</b><span>prompt · stream · sampling</span></div>
          <i aria-hidden="true">→</i>
          <div><b>TokenizeMsg</b><span>uid · text</span></div>
          <i aria-hidden="true">→</i>
          <div><b>UserMsg</b><span>uid · input_ids</span></div>
          <i aria-hidden="true">→</i>
          <div><b>Req / Batch</b><span>table_idx · cached_len</span></div>
        </div>`,
        caption: "协议对象逐步变为调度对象；KV 页位置只在 scheduler/engine 一侧出现。",
      },
    },
    {
      id: "radix-cache",
      title: "02 · Radix Cache：token 前缀到 KV 页",
      summary: "压缩前缀树把公共 token 段存为共享节点，节点 value 指向已经计算的 KV 位置。",
      paragraphs: [
        "RadixPrefixCache 的 key 是 token 序列，value 是对应的分页 KV 索引。match_prefix 从根按首个 token 或首个 page 查找子节点；只匹配到节点中间时，split_at 会把公共部分提升成父节点。匹配长度按 page_size 向下对齐，因此不能复用不完整的一页。",
        "请求使用命中的 handle 前先 lock：从命中节点向根增加 ref_count，保护正在使用的路径。释放后这些节点转为 evictable。空间不足时，evict 从未被引用的叶节点中按 timestamp 取旧节点；删除叶子后，父节点可能成为下一轮候选。",
      ],
      bullets: [
        "CacheManager.match_req 只查询 input_ids 的前 input_len - 1 个 token，为当前 forward 留出至少一个待计算 token。",
        "命中只省去 attention 的历史前缀计算；未命中后缀仍进入 prefill。",
        "前缀树管理复用关系，page table 管理逻辑请求位置到物理 KV 槽位的映射。",
      ],
      code: {
        title: "前缀查询中的节点切分",
        language: "python",
        snippet: `while prefix_len < indice_len:
    child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
    if child_node is None:
        return node, prefix_len
    node = child_node
    match_len = node.get_match_len(input_ids[prefix_len:])
    match_len = align_down(match_len, self.page_size)
    prefix_len += match_len
    if match_len != node.length:
        node = node.split_at(match_len)
        node.timestamp = tic
        return node, prefix_len`,
        sourceUrl: `${upstream}/python/minisgl/kvcache/radix_cache.py#L207-L225`,
        sourceLabel: "python/minisgl/kvcache/radix_cache.py",
        note: "官方源码节选；省略完整命中后的 timestamp 更新与 return。",
      },
      visual: {
        title: "共享前缀形成分叉",
        html: `<div class="radix-static" role="img" aria-label="两个请求共享系统提示和代码助手前缀的 Radix Tree">
          <div class="radix-root">root</div>
          <div class="radix-stem"><span>&lt;sys&gt; · 你是 · 代码 · 助手</span><small>共享 KV 页</small></div>
          <div class="radix-branches">
            <div><span>解释 · Radix · Cache</span><small>请求 A</small></div>
            <div><span>解释 · Chunk · Prefill</span><small>请求 B</small></div>
          </div>
        </div>`,
        caption: "树节点保存一段 token，不要求一个 token 对应一个节点；实际可复用边界受 page_size 约束。",
      },
    },
    {
      id: "chunked-prefill",
      title: "03 · Chunked Prefill：用 token budget 切长输入",
      summary: "调度器只把未命中的输入后缀计入本轮 token budget，剩余部分保留为 ChunkedReq。",
      paragraphs: [
        "PrefillAdder 先查询 cached_len，再计算 remain_len。chunk_size 是本轮剩余 token_budget 与 remain_len 的较小值。如果还有未处理 token，调度对象使用 ChunkedReq；它不能采样，也不会进入 decode manager。",
        "下一轮 PrefillManager 优先保留 chunked_list，并使用同一 cache_handle 和 table_idx 继续追加输入。Chunked Prefill 限制一次 forward 的 extend token 数，可让 decode batch 更快再次获得调度机会；chunk 太小则会增加 batch 准备和 kernel 发射次数。",
      ],
      bullets: [
        "预算由 max_extend_tokens 配置，并非按请求数切分。",
        "已缓存 token 不占本轮 prefill token budget。",
        "内存准入还会预留输入剩余量、输出上限和正在飞行的 decode page。",
      ],
      code: {
        title: "根据剩余预算构造普通或分块请求",
        language: "python",
        snippet: `remain_len = pending_req.input_len - cached_len
chunk_size = min(self.token_budget, remain_len)
is_chunked = chunk_size < remain_len
CLS = ChunkedReq if is_chunked else Req
self.token_budget -= chunk_size
self.reserved_size += remain_len + pending_req.output_len
_slice = slice(cached_len, cached_len + chunk_size)
device_ids = self.table_manager.token_pool[table_idx, _slice]
device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)`,
        sourceUrl: `${upstream}/python/minisgl/scheduler/prefill.py#L61-L76`,
        sourceLabel: "python/minisgl/scheduler/prefill.py",
        note: "官方源码节选；返回 Req 的字段在原文件后续行中。",
      },
      visual: {
        title: "命中前缀后只切未计算部分",
        html: `<div class="chunk-strip" role="img" aria-label="前缀命中与三个 prefill chunk">
          <span class="is-hit">cache hit · 6</span>
          <span>chunk 1 · 4</span>
          <span>chunk 2 · 4</span>
          <span>chunk 3 · 2</span>
          <b>→ decode</b>
        </div>`,
        caption: "示意：16 token 输入命中 6 个 token，chunk size 为 4，因此只执行 4、4、2 三段 prefill。",
      },
    },
    {
      id: "overlap-scheduling",
      title: "04 · Overlap Scheduling：CPU 处理与 GPU forward 重叠",
      summary: "scheduler 使用独立 CUDA stream，在启动当前 batch 后处理上一批结果。",
      paragraphs: [
        "overlap_loop 先收消息和准备 batch，再让 Engine stream 等待 scheduler stream 上的元数据拷贝，随后启动当前 forward。GPU 工作已经入队后，CPU 才调用 _process_last_data 同步上一批 next token 的异步拷贝、判断结束条件并回收资源。",
        "这种重叠隐藏的是调度、消息和结果处理的 CPU 时间，不会减少模型本身的计算量。关闭 overlap 后，normal_loop 会立刻处理刚完成的 ongoing_data，因此 CPU 和 GPU 阶段按批次串行。依赖关系仍由 stream wait、copy_done event 和请求状态约束。",
      ],
      bullets: [
        "Scheduler stream 负责元数据与非阻塞拷贝，Engine 拥有另一条 CUDA stream。",
        "当前批 forward 与上一批结果处理重叠，而不是让同一请求跳过数据依赖。",
        "prefill 与 decode 都经过 _prepare_batch；decode 可在允许的 batch size 下复用 CUDA Graph。",
      ],
      code: {
        title: "启动当前 forward，再处理上一批结果",
        language: "python",
        snippet: `forward_input = self._schedule_next_batch()
ongoing_data = None
if forward_input is not None:
    with self.engine_stream_ctx:
        self.engine.stream.wait_stream(self.stream)
        ongoing_data = (forward_input, self._forward(forward_input))

self._process_last_data(last_data)
return ongoing_data`,
        sourceUrl: `${upstream}/python/minisgl/scheduler/scheduler.py#L99-L107`,
        sourceLabel: "python/minisgl/scheduler/scheduler.py",
        note: "官方源码节选；消息接收和 blocking 判定在此前几行。",
      },
      visual: {
        title: "两批之间的重叠窗口",
        html: `<div class="overlap-static" role="img" aria-label="CPU 调度与 GPU 计算重叠时间线">
          <div class="timeline-label">CPU</div><div class="timeline-track"><span class="cpu-a">处理 A</span><span class="cpu-b">调度 B</span><span class="cpu-c">处理 B</span></div>
          <div class="timeline-label">GPU</div><div class="timeline-track"><span class="gpu-a">forward A</span><span class="gpu-b">forward B</span></div>
        </div>`,
        caption: "CPU 的“调度 B”可以位于 GPU 的“forward A”窗口内；实验台可切换串行模式比较总时间线。",
      },
    },
    {
      id: "tp-decode-attention",
      title: "05 · Tensor Parallel 与 Decode Attention",
      summary: "每个 TP rank 持有模型与 KV 的分片，decode 时每个请求只增加一个 query token 并读取历史分页 K/V。",
      paragraphs: [
        "launch_server 为每个 TP rank 启动一个 scheduler，Scheduler 内部构造本地 Engine。QKV 列并行层按 head 切输出，O projection 和 row-parallel linear 对局部结果执行 all-reduce；词表并行 embedding/LM head 也在需要时进行 all-reduce 或 all-gather。",
        "AttentionLayer 把本 rank 的 q、k、v 分开，应用 RoPE 后交给 attention backend。backend 先把当前 token 的 K/V 写到 batch.out_loc，再使用 page_table、cache_seqlens 和累计序列长度读取历史缓存。decode 的 extend_len 为 1，计算形态与一次处理多个新 token 的 prefill 不同。",
      ],
      bullets: [
        "每个 rank 的 Engine 使用 cuda:{rank}，并加载与 TP 切分规则一致的权重。",
        "KV 每页容量按本 rank 的 num_kv_heads 计算；各 rank 保持相同逻辑请求表。",
        "FlashAttention、FlashInfer 与 TRT-LLM backend 都通过统一接口接收 Batch 元数据。",
        "采样结果从 GPU 异步复制到 CPU；rank 之间的请求与结果同步由 scheduler I/O 层处理。",
      ],
      visual: {
        title: "TP=2 时的一次 decode step",
        html: `<div class="tp-map" role="img" aria-label="两个 tensor parallel rank 执行 decode attention">
          <div class="tp-input"><span>token t</span><small>同一逻辑 batch</small></div>
          <div class="tp-ranks">
            <div><b>Rank 0</b><span>Q heads 0…h/2</span><span>KV shard 0</span><em>local attention</em></div>
            <div><b>Rank 1</b><span>Q heads h/2…h</span><span>KV shard 1</span><em>local attention</em></div>
          </div>
          <div class="tp-reduce"><span>all-reduce</span><b>hidden state → LM head → sample</b></div>
        </div>`,
        caption: "head 与权重的具体切法由层类型决定；row-parallel 输出需要归约为相同 hidden state。",
      },
    },
  ],
  lab: {
    title: "Radix Cache + Chunked Prefill 调度台",
    intro: "四个请求按顺序进入调度器。调整 chunk size、缓存容量和 overlap，观察共享前缀命中、叶路径淘汰、prefill 切片以及 CPU/GPU 时间线。数值是离线机制模型，不代表项目 benchmark。",
    html: `<div class="msgl-lab" data-msgl-lab>
      <form class="msgl-controls" data-controls>
        <label>
          <span>Chunk size <output data-chunk-value>4</output> tokens</span>
          <input data-chunk type="range" min="2" max="8" step="1" value="4">
        </label>
        <label>
          <span>Cache capacity <output data-capacity-value>18</output> tokens</span>
          <input data-capacity type="range" min="8" max="28" step="1" value="18">
        </label>
        <label class="msgl-switch">
          <input data-overlap type="checkbox" checked>
          <span>Overlap scheduling</span>
        </label>
        <button type="submit">重新调度</button>
      </form>

      <div class="msgl-metrics" aria-label="调度结果摘要">
        <div><span>前缀命中</span><strong data-hit-total>0</strong><small>tokens</small></div>
        <div><span>Prefill</span><strong data-prefill-total>0</strong><small>tokens</small></div>
        <div><span>淘汰</span><strong data-evict-total>0</strong><small>tokens</small></div>
        <div><span>模型时间</span><strong data-total-time>0</strong><small>ms</small></div>
      </div>

      <section class="msgl-panel msgl-requests" aria-labelledby="msgl-request-title">
        <div class="msgl-panel-head"><h3 id="msgl-request-title">请求与 prefill chunks</h3><span>绿色 = cache hit</span></div>
        <div data-request-list></div>
      </section>

      <div class="msgl-grid">
        <section class="msgl-panel" aria-labelledby="msgl-tree-title">
          <div class="msgl-panel-head"><h3 id="msgl-tree-title">Radix tree</h3><span data-cache-usage>0 / 0</span></div>
          <div class="msgl-tree" data-tree role="img" aria-label="模拟后的 Radix Cache 前缀树"></div>
          <div class="msgl-evictions"><b>Eviction log</b><ol data-eviction-list></ol></div>
        </section>
        <section class="msgl-panel" aria-labelledby="msgl-timeline-title">
          <div class="msgl-panel-head"><h3 id="msgl-timeline-title">CPU / GPU timeline</h3><span data-overlap-state>overlap on</span></div>
          <div class="msgl-timeline" data-timeline></div>
          <div class="msgl-legend"><span><i class="is-cpu"></i>CPU schedule</span><span><i class="is-prefill"></i>GPU prefill</span><span><i class="is-decode"></i>GPU decode</span></div>
        </section>
      </div>
      <p class="msgl-live" data-live aria-live="polite"></p>
    </div>`,
  },
  takeaways: [
    "沿消息类型追踪请求：TokenizeMsg → UserMsg → Req/Batch → DetokenizeMsg → UserReply。",
    "Radix Cache 复用 token 前缀对应的 KV 页；page_size 决定可复用与可淘汰的对齐粒度。",
    "Chunked Prefill 控制单轮 extend token 数，Overlap Scheduling 控制 CPU 调度和 GPU forward 的时间重叠，两者处理不同瓶颈。",
    "Tensor Parallel 让每个 scheduler/Engine rank 持有模型、attention head 和 KV 的局部分片，必要位置通过 collective 合并。",
    "阅读入口可从 scheduler/scheduler.py 开始，再分别下钻 prefill.py、cache.py、radix_cache.py 与 attention backend。",
  ],
  sources: [
    { label: "Mini-SGLang 官方仓库", url: "https://github.com/sgl-project/mini-sglang" },
    { label: "官方系统结构文档", url: `${upstream}/docs/structures.md` },
    { label: "Scheduler 实现", url: `${upstream}/python/minisgl/scheduler/scheduler.py` },
    { label: "Prefill 调度实现", url: `${upstream}/python/minisgl/scheduler/prefill.py` },
    { label: "Radix Cache 实现", url: `${upstream}/python/minisgl/kvcache/radix_cache.py` },
    { label: "FlashAttention backend", url: `${upstream}/python/minisgl/attention/fa.py` },
    { label: "Tensor Parallel linear layers", url: `${upstream}/python/minisgl/layers/linear.py` },
  ],
};
