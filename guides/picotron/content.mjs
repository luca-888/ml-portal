export default {
  slug: "picotron",
  title: "Picotron",
  subtitle: "用一个四维进程网格实现数据、张量、流水线与上下文并行。",
  repoUrl: "https://github.com/huggingface/picotron",
  accent: "#f59e0b",
  tags: ["Distributed Training", "4D Parallelism", "Llama"],
  audience: "想从进程组、张量形状和通信位置理解大模型分布式训练的读者。",
  prerequisites: [
    "理解反向传播与梯度累积",
    "了解 Transformer 的注意力和 MLP",
    "知道 torch.distributed 的 rank、world size 与 collective",
  ],
  outcomes: [
    "把一个 global rank 定位到 DP × PP × CP × TP 网格",
    "说明四种并行分别切分数据、层、序列或层内张量",
    "识别 all-reduce、点对点传输与 ring 通信发生的位置",
    "根据 micro-batch 数估算非交错 1F1B 的 bubble",
  ],
  overview: [
    "Picotron 把 world size 写成 DP × PP × CP × TP。每个进程只改变其中一个坐标，就得到对应的 process group；四种并行因此可以组合，而不是各自维护一套 rank 编号。",
    "DP 复制模型并切 batch；TP 切层内权重与隐藏维；PP 把连续层分配给不同 stage；CP 切序列，并在注意力中沿环交换 K/V。它们节省的内存对象不同，通信频率也不同。",
    "本页从网格开始，沿着一次训练 step 追踪张量和梯度。交互实验使用与仓库一致的 `[DP, PP, CP, TP]` rank 顺序，但只做规划与可视化，不执行分布式训练。",
  ],
  architecture: {
    title: "从 world 到一次 optimizer step",
    description: "配置先形成四维 rank 网格；每个维度建立独立进程组，随后在前向、反向或 stage 边界执行通信。",
    nodes: [
      { id: "world", label: "World", detail: "world size = DP × PP × CP × TP" },
      { id: "grid", label: "4D rank grid", detail: "形状为 [DP, PP, CP, TP]" },
      { id: "dp", label: "DP group", detail: "同一模型分片上的梯度归约" },
      { id: "tp", label: "TP group", detail: "层内权重切分与 all-reduce / all-gather" },
      { id: "pp", label: "PP group", detail: "相邻 stage 传 activation 与 gradient" },
      { id: "cp", label: "CP group", detail: "序列切分后环传 K/V" },
      { id: "loss", label: "Micro-batch loss", detail: "梯度累积期间延迟 DP/CP 同步" },
      { id: "step", label: "Optimizer step", detail: "同步完成后更新本 rank 参数分片" },
    ],
    edges: [
      { from: "world", to: "grid", label: "reshape" },
      { from: "grid", to: "dp" },
      { from: "grid", to: "tp" },
      { from: "grid", to: "pp" },
      { from: "grid", to: "cp" },
      { from: "tp", to: "loss", label: "layer compute" },
      { from: "pp", to: "loss", label: "1F1B" },
      { from: "cp", to: "loss", label: "ring attention" },
      { from: "loss", to: "dp", label: "last micro-batch" },
      { from: "dp", to: "step", label: "reduced gradients" },
    ],
  },
  sections: [
    {
      id: "rank-grid",
      title: "先定位 rank，再讨论并行",
      summary: "四维坐标决定当前进程属于哪些 TP、CP、PP 与 DP 组。",
      paragraphs: [
        "Picotron 用 `torch.arange(world_size).view(dp, pp, cp, tp)` 建立网格。rank 按 TP 最快、DP 最慢的顺序递增。例如形状 [2, 2, 1, 2] 中，rank 0 的坐标是 d0/p0/c0/t0，rank 1 只改变 TP 坐标。",
        "固定三个坐标、遍历第四个坐标，就得到一种 process group。通信必须发生在对应组内；把 DP all-reduce 发到 world group 会把不同模型分片错误地混在一起。",
      ],
      bullets: [
        "TP group：固定 d、p、c，遍历 t。",
        "CP group：固定 d、p、t，遍历 c。",
        "PP group：固定 d、c、t，遍历 p。",
        "DP group：固定 p、c、t，遍历 d。",
      ],
      code: {
        title: "四维网格与四类进程组",
        language: "python",
        snippet: `assert self.world_size == tp_size * cp_size * pp_size * dp_size
self.grid = torch.arange(self.world_size).view(
    dp_size, pp_size, cp_size, tp_size
)
self.tp_group = dist.new_subgroups_by_enumeration(
    [self.grid[d, p, c, :].tolist()
     for d in range(dp_size) for p in range(pp_size) for c in range(cp_size)]
)[0]`,
        sourceUrl: "https://github.com/huggingface/picotron/blob/main/picotron/process_group_manager.py#L4-L17",
        sourceLabel: "picotron/process_group_manager.py",
        note: "摘录并换行；仓库随后用相同方式建立 CP、PP、DP、CP×DP 与 PP×DP 组。",
      },
      visual: {
        title: "坐标变化对应通信边界",
        html: `<div class="picotron-axis-map" role="img" aria-label="四维 rank 坐标与并行方式">
          <div><b>DP</b><span>batch 副本</span><code>d ↕</code></div>
          <div><b>PP</b><span>连续层</span><code>p ↕</code></div>
          <div><b>CP</b><span>sequence</span><code>c ↕</code></div>
          <div><b>TP</b><span>hidden / heads</span><code>t ↕</code></div>
        </div>`,
        caption: "一次 collective 只沿一个坐标轴移动；其余坐标保持不变。",
      },
    },
    {
      id: "data-parallel",
      title: "DP：复制参数，最后同步梯度",
      summary: "DP 增加每 step 的样本数，但每个 DP rank 仍保存同一份模型分片。",
      paragraphs: [
        "不同 DP rank 读取不同 micro-batch，执行相同的模型分片。反向传播后，同一参数的梯度在 `cp_dp_group` 上求和并除以组大小。该组同时包含 DP 与 CP 维，因为 CP rank 处理了不同 token 分片，参数更新前也要聚合这些贡献。",
        "梯度累积时，每个 micro-batch 都 all-reduce 会重复通信。Picotron 用 `require_backward_grad_sync` 只在最后一个累积步触发同步；bucket 版本把多个参数梯度合并后再通信，并等待异步 bucket 完成。",
      ],
      bullets: [
        "参数内存不会随 DP size 下降；每个副本仍持有本地模型分片。",
        "吞吐受 batch 可扩展性影响；global batch 随 DP 与累积步数增长。",
        "梯度 all-reduce 的数据量近似与本 rank 参数量相关。",
      ],
      code: {
        title: "只在需要时归约梯度",
        language: "python",
        snippet: `def _allreduce_grads(self, grad):
    if self.require_backward_grad_sync:
        dist.all_reduce(
            grad,
            op=dist.ReduceOp.SUM,
            group=pgm.process_group_manager.cp_dp_group,
        )
        grad /= pgm.process_group_manager.cp_dp_world_size
    return grad`,
        sourceUrl: "https://github.com/huggingface/picotron/blob/main/picotron/data_parallel/data_parallel.py#L41-L49",
        sourceLabel: "picotron/data_parallel/data_parallel.py",
        note: "原始 DataParallelNaive 实现摘录；训练默认包装的是带 bucket 的 DataParallelBucket。",
      },
      visual: {
        title: "DP 的复制与归约",
        html: `<div class="picotron-dp-flow" role="img" aria-label="三个数据并行副本先分别反向传播再归约梯度">
          <div><span>batch A</span><b>model shard</b><code>∇A</code></div>
          <div><span>batch B</span><b>model shard</b><code>∇B</code></div>
          <div><span>batch C</span><b>model shard</b><code>∇C</code></div>
          <strong>all-reduce → (∇A + ∇B + ∇C) / 3</strong>
        </div>`,
        caption: "DP 切的是样本，不切本组内的参数；同步点位于反向传播。",
      },
    },
    {
      id: "tensor-parallel",
      title: "TP：在一层内部切权重",
      summary: "列并行产生隐藏维分片，行并行消费这些分片并用 all-reduce 合并部分和。",
      paragraphs: [
        "Picotron 把 q/k/v、MLP 的 up/gate projection 替换为 ColumnParallelLinear，把 attention output 与 MLP down projection 替换为 RowParallelLinear。列并行按输出特征切权重，所以每个 rank 得到 `[B, S, H/TP]`；后续行并行按输入特征切权重，局部矩阵乘结果相加才得到完整 `[B, S, H]`。",
        "TP 降低单 rank 的层内参数与中间激活，但每个 Transformer block 都会发生 collective。它更依赖低延迟、高带宽链路；若模型能放进单卡，增大 TP 不一定降低 step 时间。",
      ],
      bullets: [
        "ColumnParallelLinear：权重形状从 [Hout, Hin] 变为 [Hout/TP, Hin]。",
        "RowParallelLinear：权重形状变为 [Hout, Hin/TP]，前向结果在 TP group 上 all-reduce。",
        "词表 embedding 也按 vocabulary 维切分，局部 lookup 后归约。",
      ],
      code: {
        title: "行并行的局部乘法与合并",
        language: "python",
        snippet: `self.input_size_per_partition = (
    in_features // self.tp_world_size
)
self.weight = nn.Parameter(torch.Tensor(
    self.out_features,
    self.input_size_per_partition,
))

output_parallel = F.linear(x, self.weight)
output = ReduceFromModelParallelRegion.apply(output_parallel)`,
        sourceUrl: "https://github.com/huggingface/picotron/blob/main/picotron/tensor_parallel/tensor_parallel.py#L134-L176",
        sourceLabel: "picotron/tensor_parallel/tensor_parallel.py",
        note: "从 RowParallelLinear 摘录并合并相邻代码；ReduceFromModelParallelRegion 在 TP group 上执行 all-reduce。",
      },
      visual: {
        title: "列切分接行切分",
        html: `<div class="picotron-tp-flow" role="img" aria-label="张量并行先按输出维切分再按输入维合并">
          <div class="picotron-tp-full">X [B,S,H]</div>
          <div class="picotron-tp-shards"><span>Wq₀ → H/T</span><span>Wq₁ → H/T</span><span>Wq₂ → H/T</span><span>Wq₃ → H/T</span></div>
          <div class="picotron-tp-arrow">局部 attention / MLP ↓</div>
          <div class="picotron-tp-shards"><span>partial₀</span><span>partial₁</span><span>partial₂</span><span>partial₃</span></div>
          <div class="picotron-tp-full">all-reduce → Y [B,S,H]</div>
        </div>`,
        caption: "示意 TP=4。隐藏维分片存在于成对的 column/row projection 之间。",
      },
    },
    {
      id: "pipeline-parallel",
      title: "PP：切层，并用 micro-batch 填充流水线",
      summary: "每个 stage 只保存一段连续层，1F1B 用 warmup、稳态交替与 cooldown 控制激活存活时间。",
      paragraphs: [
        "PipelineParallel 尽量均匀地把 decoder layers 分配到 PP stages。第一 stage 保留 embedding，最后一 stage 保留 final norm 与 projection；相邻 stage 前向发送 activation，反向发送 activation gradient。",
        "1F1B 开始时，靠前 stage 需要多做若干 forward 才能让后续 stage 有工作；稳态中每个 stage 交替执行一个 forward 与一个 backward；末尾再清空剩余 backward。对非交错、阶段耗时相同的模型，bubble 近似 `(PP-1)/(micro_batches+PP-1)`。增加 micro-batch 可降低比例，但会改变 global batch、调度开销与激活存储。",
      ],
      bullets: [
        "PP 降低每 rank 的层参数量，但 stage 边界必须传 `[micro_batch, sequence/CP, hidden]`。",
        "层数不能均分时，前面的 stage 多分一层；算子耗时不等时仍可能负载不均。",
        "Picotron 的注释指出尚未实现 output tensor deallocation，warmup 激活仍占内存。",
      ],
      code: {
        title: "1F1B 的 warmup 数量",
        language: "python",
        snippet: `num_warmup_microbatches = min(
    pgm.process_group_manager.pp_world_size
    - pgm.process_group_manager.pp_rank - 1,
    data_loader.grad_acc_steps,
)
num_microbatches_remaining = (
    data_loader.grad_acc_steps - num_warmup_microbatches
)`,
        sourceUrl: "https://github.com/huggingface/picotron/blob/main/picotron/pipeline_parallel/pipeline_parallel.py#L129-L150",
        sourceLabel: "picotron/pipeline_parallel/pipeline_parallel.py",
        note: "原始 1F1B 调度摘录并换行；后续代码依次执行 warmup、steady state 与 cooldown。",
      },
      visual: {
        title: "stage 边界上的数据",
        html: `<div class="picotron-pp-flow" role="img" aria-label="四个流水线阶段之间前向传激活反向传梯度">
          <div><b>P0</b><span>embed · L0–5</span></div><i>activation →<br>← gradient</i>
          <div><b>P1</b><span>L6–11</span></div><i>activation →<br>← gradient</i>
          <div><b>P2</b><span>L12–17</span></div><i>activation →<br>← gradient</i>
          <div><b>P3</b><span>L18–23 · head</span></div>
        </div>`,
        caption: "示意 24 层、PP=4。层切分节省参数内存，stage 通信保留完整 hidden 维。",
      },
    },
    {
      id: "context-parallel",
      title: "CP：切序列，在注意力中环传 K/V",
      summary: "每个 CP rank 只持有一段 query 与序列位置，再逐步接收其他 rank 的 K/V block。",
      paragraphs: [
        "输入序列长度必须能被 CP size 整除。每个 rank 的 activation 从 `[B, S, H]` 变为 `[B, S/CP, H]`，RoPE 的 cos/sin 也按 CP rank 取连续区间。注意力仍需要看到允许访问的全部 key/value，因此仅切序列并不能消除通信。",
        "Picotron 的 RingAttentionFunc 保留本地 Q，让 K/V 沿 CP ring 传递。每一步计算一个 block attention，再用 log-sum-exp 状态合并局部结果。因果注意力只计算满足 `step <= rank` 的 block；反向阶段还要环传 K/V 及其梯度。",
      ],
      bullets: [
        "CP 主要降低长序列激活与 attention 中间量的单 rank 占用。",
        "通信量随序列与 head 维增长，ring 需要 CP-1 轮点对点交换。",
        "TP 切 hidden/head，CP 切 sequence；二者组合时本地逻辑 shape 为 `[B, S/CP, H/TP]`。",
      ],
      code: null,
      visual: {
        title: "K/V 绕环，Q 留在本地",
        html: `<div class="picotron-cp-ring" role="img" aria-label="四个上下文并行 rank 组成 K V 通信环">
          <div style="--cp-angle:0deg"><span>C0</span><small>Q₀ · K/V₀</small></div>
          <div style="--cp-angle:90deg"><span>C1</span><small>Q₁ · K/V₁</small></div>
          <div style="--cp-angle:180deg"><span>C2</span><small>Q₂ · K/V₂</small></div>
          <div style="--cp-angle:270deg"><span>C3</span><small>Q₃ · K/V₃</small></div>
          <strong>K/V ↻</strong>
        </div>`,
        caption: "每轮只传一个 K/V block；在线 softmax 把多个 block 的输出合并为本地 Q 的结果。",
      },
    },
  ],
  lab: {
    title: "4D 并行拓扑规划器",
    intro: "设置 GPU 总数和 DP/TP/PP/CP。规划器校验乘积，按 Picotron 的 rank 顺序画出拓扑、当前 rank 的四类进程组、1F1B 时间轴与本地张量 shape。",
    html: `<div class="picotron-lab" data-picotron-lab>
      <div class="picotron-controls" aria-label="四维并行配置">
        <label>GPU 总数<input type="number" min="1" max="64" step="1" value="16" data-picotron-input="gpus"></label>
        <label>DP<input type="number" min="1" max="64" step="1" value="2" data-picotron-input="dp"></label>
        <label>TP<input type="number" min="1" max="64" step="1" value="2" data-picotron-input="tp"></label>
        <label>PP<input type="number" min="1" max="16" step="1" value="2" data-picotron-input="pp"></label>
        <label>CP<input type="number" min="1" max="16" step="1" value="2" data-picotron-input="cp"></label>
        <label>micro-batches<input type="number" min="1" max="32" step="1" value="8" data-picotron-input="micro"></label>
        <label>序列长度<input type="number" min="128" max="131072" step="128" value="4096" data-picotron-input="sequence"></label>
        <label>隐藏维度<input type="number" min="128" max="65536" step="128" value="4096" data-picotron-input="hidden"></label>
      </div>

      <div class="picotron-status" data-picotron-status></div>
      <p class="picotron-live" aria-live="polite" aria-atomic="true" data-picotron-live></p>

      <section class="picotron-panel" aria-labelledby="picotron-topology-title">
        <div class="picotron-panel-head">
          <div><span>01</span><h3 id="picotron-topology-title">rank topology</h3></div>
          <label>检查 rank<select data-picotron-rank aria-label="选择要检查的 global rank"></select></label>
        </div>
        <div class="picotron-topology" data-picotron-topology></div>
        <div class="picotron-groups" data-picotron-groups></div>
      </section>

      <section class="picotron-panel" aria-labelledby="picotron-pipeline-title">
        <div class="picotron-panel-head">
          <div><span>02</span><h3 id="picotron-pipeline-title">1F1B pipeline</h3></div>
          <output data-picotron-bubble></output>
        </div>
        <div class="picotron-legend" aria-hidden="true"><span><i class="is-forward"></i>Forward</span><span><i class="is-backward"></i>Backward</span><span><i class="is-idle"></i>Bubble</span></div>
        <div class="picotron-timeline-wrap"><div class="picotron-timeline" data-picotron-timeline></div></div>
      </section>

      <section class="picotron-panel" aria-labelledby="picotron-shape-title">
        <div class="picotron-panel-head"><div><span>03</span><h3 id="picotron-shape-title">tensor shape</h3></div></div>
        <div class="picotron-shapes" data-picotron-shapes></div>
      </section>
    </div>`,
  },
  takeaways: [
    "先满足 world size = DP × PP × CP × TP，再为每个 rank 确定四维坐标和进程组。",
    "DP 切 batch 并同步梯度；TP 切层内矩阵；PP 切层并传激活；CP 切序列并环传 K/V。",
    "TP 与 CP 缩小不同的张量轴，PP 缩小层数；DP 通常不降低参数内存。",
    "1F1B 的 bubble 随 PP 增大、随 micro-batch 增多而下降，但更多 micro-batch 会影响 batch 与调度成本。",
    "组合并行的选择取决于模型是否能放入单卡、序列长度、节点内互联和跨节点带宽。",
  ],
  sources: [
    { label: "Picotron README 与运行示例", url: "https://github.com/huggingface/picotron/blob/main/README.md" },
    { label: "四维进程组", url: "https://github.com/huggingface/picotron/blob/main/picotron/process_group_manager.py" },
    { label: "Tensor Parallel", url: "https://github.com/huggingface/picotron/blob/main/picotron/tensor_parallel/tensor_parallel.py" },
    { label: "Tensor Parallel 通信", url: "https://github.com/huggingface/picotron/blob/main/picotron/tensor_parallel/tp_communications.py" },
    { label: "Pipeline Parallel 与 1F1B", url: "https://github.com/huggingface/picotron/blob/main/picotron/pipeline_parallel/pipeline_parallel.py" },
    { label: "Data Parallel", url: "https://github.com/huggingface/picotron/blob/main/picotron/data_parallel/data_parallel.py" },
    { label: "Context Parallel 与 Ring Attention", url: "https://github.com/huggingface/picotron/blob/main/picotron/context_parallel/context_parallel.py" },
    { label: "训练入口与并行组合顺序", url: "https://github.com/huggingface/picotron/blob/main/train.py" },
  ],
};
