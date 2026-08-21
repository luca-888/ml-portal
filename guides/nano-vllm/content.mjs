const upstream = "https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8";

export default {
  slug: "nano-vllm",
  title: "Nano-vLLM",
  subtitle: "沿一次 generate 请求理解连续批处理、Paged KV Cache 与张量并行。",
  repoUrl: "https://github.com/GeeeekExplorer/nano-vllm",
  accent: "#635bff",
  tags: ["LLM Inference", "Paged KV Cache", "Continuous Batching", "CUDA Graph", "Tensor Parallel"],
  audience: "已掌握 Python、PyTorch 张量操作和 Transformer 推理流程，希望继续阅读推理引擎代码的学习者。",
  prerequisites: [
    "知道自回归生成由一次 prefill 和多次 decode 组成",
    "理解 Attention 中 K、V 会随上下文增长并被重复读取",
    "了解 CUDA kernel launch 与 torch.distributed 的基本概念",
  ],
  outcomes: [
    "从 LLM.generate 追踪 Sequence 的 WAITING、RUNNING、FINISHED 状态",
    "说明 Scheduler 如何在 prefill、decode 与重算式抢占之间选择",
    "用 block table 把逻辑 token 位置换算为物理 KV slot",
    "解释 CUDA Graph 的静态缓冲区和张量并行中的切分与通信",
  ],
  overview: [
    "Nano-vLLM 把大语言模型推理引擎压缩到少量模块：LLMEngine 接收请求，Scheduler 选择本轮序列，BlockManager 管理 KV block，ModelRunner 准备张量并执行模型。阅读时可把每次 engine.step 看成一个离散时刻。",
    "页面基于官方仓库主分支提交 bb823b3。交互实验保留调度与页分配的因果关系，不执行真实模型，也不模拟 prefix cache 的内容哈希复用。",
  ],
  architecture: {
    title: "一次生成请求的控制路径",
    description: "实线表示一次 step 内的调用；队列和 block table 跨 step 保留状态。",
    nodes: [
      { id: "api", label: "LLM.generate", detail: "编码 prompt，逐个调用 add_request，并循环执行 step。" },
      { id: "sequence", label: "Sequence", detail: "保存 token、采样参数、状态、缓存进度和 block_table。" },
      { id: "waiting", label: "waiting", detail: "尚未完成 prefill，或因显存不足被抢占的序列。" },
      { id: "scheduler", label: "Scheduler", detail: "prefill 优先；没有可执行 prefill 时组成 decode batch。" },
      { id: "blocks", label: "BlockManager", detail: "分配、引用、哈希和释放物理 KV blocks。" },
      { id: "runner", label: "ModelRunner", detail: "构造 input_ids、positions、slot_mapping 与 block_tables。" },
      { id: "attention", label: "Paged Attention", detail: "把 K/V 写入物理 slot，并通过 block table 读取上下文。" },
      { id: "sampler", label: "Sampler", detail: "rank 0 根据 logits 与 temperature 产生 token。" },
      { id: "finished", label: "FINISHED", detail: "达到 EOS 或 max_tokens 后释放 blocks 并返回输出。" },
    ],
    edges: [
      { from: "api", to: "sequence", label: "构造" },
      { from: "sequence", to: "waiting", label: "入队" },
      { from: "waiting", to: "scheduler", label: "选择" },
      { from: "scheduler", to: "blocks", label: "申请/扩容" },
      { from: "scheduler", to: "runner", label: "seqs + phase" },
      { from: "blocks", to: "runner", label: "block_table" },
      { from: "runner", to: "attention", label: "slot mapping" },
      { from: "attention", to: "sampler", label: "hidden states" },
      { from: "sampler", to: "finished", label: "append token" },
      { from: "sampler", to: "scheduler", label: "下一 step" },
    ],
  },
  sections: [
    {
      id: "request-lifecycle",
      title: "1. generate 不是一次模型调用",
      summary: "每个 prompt 变成一个 Sequence；generate 反复执行 step，直到 waiting 与 running 都为空。",
      paragraphs: [
        "add_request 先用 tokenizer 得到 token ids，再创建 Sequence 并放入 waiting。Sequence 同时保存 num_prompt_tokens、num_cached_tokens、num_scheduled_tokens、block_table 和采样参数。",
        "一次 step 依次完成 schedule、ModelRunner.run 和 postprocess。只有达到 EOS（且未忽略）或 completion token 数等于 max_tokens，状态才变为 FINISHED；此时 block table 对应的物理页被释放。",
      ],
      bullets: [
        "WAITING：等待首次或重算 prefill。",
        "RUNNING：prompt 已进入 KV Cache，可以参加逐 token decode。",
        "FINISHED：输出完成，已从 running 移除且不再持有 KV blocks。",
      ],
      visual: {
        title: "请求 A 的 step 时序",
        html: `<div class="nv-timing" role="img" aria-label="请求从到达到完成的时序图">
          <div class="nv-timing__axis"><span>t0</span><span>t1</span><span>t2</span><span>t3</span><span>t4</span></div>
          <div class="nv-timing__lane"><b>队列</b><span data-state="wait">WAITING</span><span data-state="run">RUNNING</span><span data-state="run">RUNNING</span><span data-state="done">FINISHED</span></div>
          <div class="nv-timing__lane"><b>计算</b><span>到达</span><span data-phase="prefill">prefill 8 tok</span><span data-phase="decode">decode 1 tok</span><span data-phase="decode">decode 1 tok</span></div>
          <div class="nv-timing__lane"><b>KV</b><span>0 page</span><span>2 pages</span><span>3 pages</span><span>释放</span></div>
        </div>`,
        caption: "prefill 完成时会采样第一个 completion token；后续 decode step 把上一个 token 写入 KV，再采样下一个 token。",
      },
    },
    {
      id: "continuous-batching",
      title: "2. 调度器在 token 边界重组 batch",
      summary: "调度以 step 为单位：有可执行的 waiting 请求时先做 prefill，否则从 running 组成 decode batch。",
      paragraphs: [
        "prefill 受 max_num_batched_tokens 和 max_num_seqs 约束。当前实现只允许当轮第一条序列使用 chunked prefill；如果后续序列的剩余 prompt 放不进 token budget，本轮停止继续取 waiting。",
        "decode 时每条序列只调度一个 token。若下一个 KV token 需要新 block 而空闲 block 不足，调度器从 running 尾部抢占序列；抢占会清空该序列 block table、将 is_prefill 设回 True，并放回 waiting 头部，因此恢复方式是重算而不是换出到 CPU。",
      ],
      bullets: [
        "连续批处理不是让一个固定 batch 一直跑完；完成的请求可离开，新请求可在后续 step 加入。",
        "prefill 优先减少新请求开始服务前的等待，但会改变 decode batch 的连续性。",
        "重算式抢占释放显存直接，但被抢占请求需要再次执行 prompt。",
      ],
      code: {
        title: "decode 扩容与重算式抢占",
        language: "python",
        snippet: `while self.running and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.running.popleft()
    while not self.block_manager.can_append(seq):
        if self.running:
            self.preempt(self.running.pop())
        else:
            self.preempt(seq)
            break
    else:
        seq.num_scheduled_tokens = 1
        seq.is_prefill = False
        self.block_manager.may_append(seq)
        scheduled_seqs.append(seq)`,
        sourceUrl: `${upstream}/nanovllm/engine/scheduler.py#L57-L70`,
        sourceLabel: "nanovllm/engine/scheduler.py",
        note: "官方源码原文；截取 decode 调度分支。",
      },
      visual: {
        title: "prefill-first 决策",
        html: `<div class="nv-decision" role="img" aria-label="prefill 优先调度决策图">
          <div class="nv-decision__question">waiting 头部可分配 pages？</div>
          <div class="nv-decision__branch"><span><b>是</b>按 token budget 组成 prefill batch</span><span><b>否</b>尝试 running decode</span></div>
          <div class="nv-decision__branch"><span>prompt 未完成 → 保留 WAITING</span><span>缺页 → 抢占尾部 RUNNING</span></div>
        </div>`,
        caption: "实验中减少物理 pages，可观察 decode 为新页腾空间时发生的抢占和重算。",
      },
    },
    {
      id: "paged-kv-cache",
      title: "3. block table 把连续上下文映射到离散显存页",
      summary: "Sequence 只记录逻辑顺序，BlockManager 决定每个逻辑 block 对应哪个物理 block id。",
      paragraphs: [
        "在默认配置中，kvcache_block_size 为 256。Sequence.num_blocks 对 token 数向上取整；block_table[i] 给出逻辑第 i 页所在的物理 block。物理页不要求相邻，因此多个长度不同的请求可以共享同一个 block 池。",
        "BlockManager 还会对完整 token block 做链式 xxhash。can_allocate 从 prompt 开头检查相同内容，命中时可复用已有 block 并增加 ref_count；最后一个未满 block 不参与这段前缀匹配。",
        "ModelRunner 把逻辑位置换算为 slot_mapping = block_id × block_size + 页内偏移。Triton store_kvcache kernel 按 slot_mapping 写 K/V；decode attention 再结合 block_tables 和 context_lens 读取上下文。",
      ],
      bullets: [
        "block table 是每条序列的逻辑页表；KV tensor 是所有序列共享的物理页池。",
        "释放序列时 ref_count 逐页减一，归零的物理页回到 free_block_ids。",
        "页越大，页表越短但末页内部碎片可能越多；实验用较小页尺寸便于观察。",
      ],
      code: {
        title: "分配前先查找可复用的完整前缀 blocks",
        language: "python",
        snippet: `def can_allocate(self, seq: Sequence) -> int:
    h = -1
    num_cached_blocks = 0
    num_new_blocks = seq.num_blocks
    for i in range(seq.num_blocks - 1):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            break
        num_cached_blocks += 1
        if block_id in self.used_block_ids:
            num_new_blocks -= 1
    if len(self.free_block_ids) < num_new_blocks:
        return -1
    return num_cached_blocks`,
        sourceUrl: `${upstream}/nanovllm/engine/block_manager.py#L58-L73`,
        sourceLabel: "nanovllm/engine/block_manager.py",
        note: "官方源码原文。交互实验不模拟哈希命中，只模拟独占物理页的分配与释放。",
      },
      visual: {
        title: "两条序列的逻辑页表",
        html: `<div class="nv-page-map" role="img" aria-label="逻辑页映射到非连续物理页">
          <div class="nv-page-map__row"><b>A logical</b><span>L0</span><span>L1</span><i>→</i><span data-page="a">P3</span><span data-page="a">P0</span></div>
          <div class="nv-page-map__row"><b>B logical</b><span>L0</span><span>L1</span><span>L2</span><i>→</i><span data-page="b">P1</span><span data-page="b">P5</span><span data-page="b">P2</span></div>
        </div>`,
        caption: "Attention 按逻辑顺序遍历 block table；物理 block id 无需连续。",
      },
    },
    {
      id: "prefill-decode",
      title: "4. prefill 与 decode 使用不同形状和算子",
      summary: "prefill 展平多个 token 段，decode 每条序列输入一个 token；两者共享同一组 KV pages。",
      paragraphs: [
        "prepare_prefill 为每条序列计算 start、end、cu_seqlens_q/k 和 slot_mapping。存在 prefix cache 命中时，query 只含未缓存 token，而 key/value 的逻辑长度仍覆盖已缓存前缀，因此需要 block_tables。",
        "prepare_decode 为每条序列只放入 last_token，并提供 position、context_len、slot_mapping 和补齐后的 block_tables。Attention 在 prefill 分支调用 flash_attn_varlen_func，在 decode 分支调用 flash_attn_with_kvcache。",
      ],
      bullets: [
        "prefill 的工作量随 prompt token 数增长，适合变长批处理元数据。",
        "decode 的 query 长度固定为 1，但每一步要读取逐渐增长的 KV 上下文。",
        "slot_mapping 决定当前 K/V 写到哪里；block_tables 决定历史 K/V 从哪里读。",
      ],
      visual: {
        title: "同一 KV 池上的两种阶段",
        html: `<div class="nv-phases" role="img" aria-label="prefill 与 decode 数据形状对比">
          <div><b>Prefill</b><span class="nv-token nv-token--wide">A0 A1 A2 A3 A4</span><span class="nv-token">B0 B1</span><small>变长 query → 多个 slot 写入</small></div>
          <div><b>Decode</b><span class="nv-token">A5</span><span class="nv-token">B2</span><small>每序列 1 token → 读取各自 block table</small></div>
        </div>`,
        caption: "Scheduler 一次只返回一种 phase，因此 ModelRunner 可选择对应的输入准备和 attention 路径。",
      },
    },
    {
      id: "cuda-graph",
      title: "5. CUDA Graph 把 decode 的 launch 序列预先捕获",
      summary: "项目仅对满足条件的 decode 使用 CUDA Graph，并用预分配张量承接每一步变化的元数据。",
      paragraphs: [
        "初始化时 capture_cudagraph 为一组 batch size 捕获图：1、2、4、8，之后从 16 起按 16 递增到 max_num_seqs（上限 512）。不同 batch size 共用 graph pool。",
        "运行时选择不小于真实 batch size 的最小已捕获图，把 input_ids、positions、slot_mapping、context_lens 和 block_tables 复制进静态缓冲区，再 replay。slot_mapping 的未使用位置填 -1，context_lens 的未使用位置清零。",
        "prefill、enforce_eager=True 或 batch size 大于 512 时直接走 eager。Graph 依赖预分配 shape：max_model_len 和最大 block-table 宽度会影响捕获缓冲区。",
      ],
      bullets: [
        "目标是减少每个 decode step 的 Python/CUDA launch 开销，而不是减少 Attention 的计算量。",
        "图的 batch size 可大于真实 batch；无效槽位必须用不会写 KV 的值填充。",
        "enforce_eager 是对照和排错开关。",
      ],
      code: {
        title: "把动态元数据写入静态 Graph 缓冲区",
        language: "python",
        snippet: `bs = input_ids.size(0)
context = get_context()
graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
graph_vars = self.graph_vars
graph_vars["input_ids"][:bs] = input_ids
graph_vars["positions"][:bs] = positions
graph_vars["slot_mapping"].fill_(-1)
graph_vars["slot_mapping"][:bs] = context.slot_mapping
graph_vars["context_lens"].zero_()
graph_vars["context_lens"][:bs] = context.context_lens
graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
graph.replay()
return self.model.compute_logits(graph_vars["outputs"][:bs])`,
        sourceUrl: `${upstream}/nanovllm/engine/model_runner.py#L200-L212`,
        sourceLabel: "nanovllm/engine/model_runner.py",
        note: "官方源码原文；外层条件已在正文说明。",
      },
      visual: {
        title: "真实 batch 映射到捕获 shape",
        html: `<div class="nv-graph-buckets" role="img" aria-label="CUDA Graph batch size 分桶">
          <span>1 → G1</span><span>3 → G4</span><span>7 → G8</span><span>11 → G16</span><span>31 → G32</span>
        </div>`,
        caption: "选择规则是 next(x for x in graph_bs if x >= bs)，因此 bs=11 重放 G16。",
      },
    },
    {
      id: "tensor-parallel",
      title: "6. 张量并行让每个 rank 保存一部分权重与 KV heads",
      summary: "rank 0 负责调度、广播调用和采样；每个 rank 执行自己持有的权重分片。",
      paragraphs: [
        "LLMEngine 为 rank 1…N-1 启动 ModelRunner 进程，rank 0 也创建一个 ModelRunner。调用名和 Sequence 参数通过 SharedMemory 写入，Event 唤醒其他 rank；模型计算用 NCCL process group 协作。",
        "ColumnParallelLinear 沿输出维切权重，局部输出可直接进入下一层的 RowParallelLinear；RowParallelLinear 沿输入维切权重，局部结果通过 all_reduce 相加。Q/K/V heads 和每层 KV cache 的 num_kv_heads 同样除以 world_size。",
        "词表 embedding 在各 rank 切分后用 all_reduce 合成 hidden states；LM head 的局部 logits gather 到 rank 0，只有 rank 0 进行采样并返回 token ids。",
      ],
      bullets: [
        "tensor_parallel_size 必须在 1 到 8 之间，并且相关维度要能被 world size 整除。",
        "每个 rank 持有相同逻辑 block id 的本地 KV-head 分片。",
        "通信点位于 row-parallel 输出、vocab embedding 和最终 logits 汇集。",
      ],
      code: {
        title: "RowParallelLinear 合并各 rank 的部分和",
        language: "python",
        snippet: `def forward(self, x: torch.Tensor) -> torch.Tensor:
    y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
    if self.tp_size > 1:
        dist.all_reduce(y)
    return y`,
        sourceUrl: `${upstream}/nanovllm/layers/linear.py#L152-L156`,
        sourceLabel: "nanovllm/layers/linear.py",
        note: "官方源码原文。",
      },
      visual: {
        title: "两卡 MLP/Attention 线性层的数据流",
        html: `<div class="nv-tp" role="img" aria-label="两路张量并行数据流">
          <div class="nv-tp__input">输入 x</div>
          <div class="nv-tp__ranks"><span><b>rank 0</b>列切权重 → 局部通道</span><span><b>rank 1</b>列切权重 → 局部通道</span></div>
          <div class="nv-tp__ranks"><span>行切权重 → 部分和 y₀</span><span>行切权重 → 部分和 y₁</span></div>
          <div class="nv-tp__reduce">all_reduce(y₀ + y₁)</div>
        </div>`,
        caption: "列并行产生分片激活，行并行把各分片对输出的贡献相加。",
      },
    },
  ],
  lab: {
    title: "连续批处理与 KV page 分配模拟器",
    intro: "设置三个请求的到达 step、prompt/output token 数、物理 page 数和每步 token budget。逐步运行可观察 prefill-first、waiting/running/finished 队列、逻辑页表、物理页利用率与重算式抢占。实验页尺寸可调，默认值刻意缩小以显示页边界。",
    html: `<div class="nv-lab" data-nv-lab>
      <div class="nv-lab__setup" aria-label="模拟参数">
        <div class="nv-lab__global">
          <label>物理 pages <input type="number" min="3" max="24" value="8" data-nv-pages></label>
          <label>page size <input type="number" min="2" max="16" value="4" data-nv-page-size></label>
          <label>每步 token budget <input type="number" min="1" max="32" value="6" data-nv-budget></label>
        </div>
        <div class="nv-lab__requests">
          <fieldset data-nv-request="A"><legend><i data-color="a"></i>请求 A</legend><label>到达 <input type="number" min="0" max="20" value="0" data-field="arrival"></label><label>prompt <input type="number" min="1" max="64" value="7" data-field="prompt"></label><label>output <input type="number" min="1" max="32" value="5" data-field="output"></label></fieldset>
          <fieldset data-nv-request="B"><legend><i data-color="b"></i>请求 B</legend><label>到达 <input type="number" min="0" max="20" value="1" data-field="arrival"></label><label>prompt <input type="number" min="1" max="64" value="5" data-field="prompt"></label><label>output <input type="number" min="1" max="32" value="4" data-field="output"></label></fieldset>
          <fieldset data-nv-request="C"><legend><i data-color="c"></i>请求 C</legend><label>到达 <input type="number" min="0" max="20" value="3" data-field="arrival"></label><label>prompt <input type="number" min="1" max="64" value="9" data-field="prompt"></label><label>output <input type="number" min="1" max="32" value="3" data-field="output"></label></fieldset>
        </div>
        <div class="nv-lab__actions">
          <button type="button" data-nv-step>下一 step</button>
          <button type="button" data-nv-play>连续运行</button>
          <button type="button" data-nv-reset>应用参数并重置</button>
          <span>快捷键：Space 单步，R 重置</span>
        </div>
      </div>

      <div class="nv-lab__status" aria-label="当前调度状态">
        <div><small>STEP</small><strong data-nv-clock>0</strong></div>
        <div><small>PHASE</small><strong data-nv-phase>等待开始</strong></div>
        <div><small>SCHEDULED</small><strong data-nv-scheduled>—</strong></div>
        <div class="nv-lab__meter" style="--util: 0%"><small>PAGE UTIL</small><strong data-nv-util>0%</strong></div>
      </div>

      <div class="nv-lab__queues">
        <section><h4>WAITING <span data-nv-wait-count>0</span></h4><div data-nv-waiting class="nv-queue"></div></section>
        <section><h4>RUNNING <span data-nv-run-count>0</span></h4><div data-nv-running class="nv-queue"></div></section>
        <section><h4>FINISHED <span data-nv-finish-count>0</span></h4><div data-nv-finished class="nv-queue"></div></section>
      </div>

      <section class="nv-lab__timeline" aria-label="请求进度">
        <div class="nv-panel-title"><h4>请求进度</h4><span>prompt / output</span></div>
        <div data-nv-timeline></div>
      </section>

      <section class="nv-lab__memory" aria-label="KV page table">
        <div class="nv-panel-title"><h4>KV page pool</h4><span data-nv-memory-note>每页 4 tokens</span></div>
        <div class="nv-memory-layout">
          <div class="nv-physical-pages" data-nv-physical></div>
          <div class="nv-page-tables" data-nv-tables></div>
        </div>
      </section>

      <section class="nv-lab__events">
        <div class="nv-panel-title"><h4>事件</h4><span>最新在上</span></div>
        <ol data-nv-events></ol>
      </section>
      <p class="nv-sr-live" aria-live="polite" aria-atomic="true" data-nv-live></p>
    </div>`,
  },
  takeaways: [
    "把 generate 拆成 step，才能看见连续批处理在请求到达、完成和抢占时如何改变 batch。",
    "Sequence.block_table 是控制面元数据；slot_mapping 和 block_tables 把它转换成 Attention 使用的物理地址。",
    "prefill 与 decode 的 shape 和瓶颈不同，Nano-vLLM 为它们选择不同的 FlashAttention 调用路径。",
    "CUDA Graph 通过静态缓冲区复用 decode launch；张量并行通过权重切分和 collective 通信组合结果。",
  ],
  sources: [
    { label: "官方仓库（源码快照 bb823b3）", url: "https://github.com/GeeeekExplorer/nano-vllm/tree/bb823b3e06983d71485a8e1f23715ebd87d98ef8" },
    { label: "LLMEngine：请求入口与生成循环", url: `${upstream}/nanovllm/engine/llm_engine.py` },
    { label: "Sequence：状态与 token/cache 元数据", url: `${upstream}/nanovllm/engine/sequence.py` },
    { label: "Scheduler：prefill、decode 与抢占", url: `${upstream}/nanovllm/engine/scheduler.py` },
    { label: "BlockManager：Paged KV blocks 与 prefix cache", url: `${upstream}/nanovllm/engine/block_manager.py` },
    { label: "ModelRunner：KV Cache、输入准备与 CUDA Graph", url: `${upstream}/nanovllm/engine/model_runner.py` },
    { label: "Attention：KV 写入与 FlashAttention 路径", url: `${upstream}/nanovllm/layers/attention.py` },
    { label: "Linear：张量并行线性层", url: `${upstream}/nanovllm/layers/linear.py` },
    { label: "Embedding/LM Head：词表切分与 logits gather", url: `${upstream}/nanovllm/layers/embed_head.py` },
  ],
};
