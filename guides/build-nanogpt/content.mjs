const repo = "https://github.com/karpathy/build-nanogpt";
const at = (ref, path = "train_gpt2.py") => `${repo}/blob/${ref}/${path}`;

const refs = {
  loader: "631f7d6f080effc4342b5e2f2f41879642c76fd1",
  weights: "9ac321e299f06c0414bd577fd2776f4562283a8a",
  flash: "7ee630c8aa426cc9d22cff30c1b3480a24e016a1",
  ddpFix: "f6d194b90f63ee6b7159974078ea60221c172617",
  eval: "8018ed2c82c78a0600b2f7c66b1e606a93c60584",
  checkpoint: "efedfacd755a507dd1031e251dc0375e6a11d1cc",
};

export default {
  slug: "build-nanogpt",
  title: "build-nanogpt",
  subtitle: "沿提交顺序从 GPT-2 数据批次构建到多卡训练与评估。",
  repoUrl: repo,
  accent: "#d9482b",
  tags: ["GPT-2", "PyTorch", "DDP", "Training loop"],
  audience: "理解 Transformer 结构，想按代码出现顺序读懂 GPT-2 训练循环的读者。",
  prerequisites: [
    "知道 token、logits、交叉熵与反向传播",
    "能阅读 PyTorch Module、optimizer 与张量 shape",
    "了解 GPU 显存限制和数据并行的基本目的",
  ],
  outcomes: [
    "解释输入和目标为什么只相差一个 token",
    "追踪 GPT-2 权重如何载入自写的 Transformer",
    "区分 mixed precision、torch.compile 与 Flash Attention 的作用层级",
    "计算 global token batch，并指出 DDP 梯度同步发生在哪个 micro-step",
    "区分训练损失、验证损失、HellaSwag、生成样本与 checkpoint",
  ],
  overview: [
    "build-nanogpt 不是把最终训练脚本拆成若干文件，而是让同一个 train_gpt2.py 随提交逐步生长。官方 README 说明这些提交被刻意保留为可顺序阅读的课程：先让模型前向与生成成立，再加入 token batch、损失、训练循环、数据加载、GPT-2 初始化、性能优化、多卡训练和评估。",
    "这种顺序把依赖关系显式化。没有稳定的 (B, T) 输入，就无法讨论模型吞吐；没有单卡训练 step，就不需要 gradient accumulation；没有明确的 global batch，DDP 也没有可保持的训练语义。每项机制都在前一版本已经可运行后出现。",
    "页面以官方 2024 年 6 月的提交和对应文件快照为依据。交互实验只计算 shape、token 数与同步位置，不运行 PyTorch，也不估算吞吐或训练时间。",
  ],
  architecture: {
    title: "一个 optimizer step 的数据与同步路径",
    description: "从左到右查看 token shard 如何形成错位目标，经过 Transformer 与损失后累积梯度；DDP 只在最后一个 micro-step 同步，再更新参数。",
    nodes: [
      { id: "shards", label: "FineWeb shards", detail: "train/val 的 NumPy token shard；DataLoaderLite 按 rank 选择起点并跨 shard 读取。" },
      { id: "loader", label: "DataLoaderLite", detail: "每个 rank、每个 micro-step 读取 B×T+1 个连续 token。" },
      { id: "batch", label: "x / y [B,T]", detail: "x 使用 buf[:-1]，y 使用 buf[1:]；每个位置预测下一个 token。" },
      { id: "gpt", label: "GPT-2 forward", detail: "token/position embedding 经 12 个 Transformer block，输出词表 logits。" },
      { id: "loss", label: "Cross entropy", detail: "展平 B×T 个位置，与目标 token 计算平均交叉熵。" },
      { id: "accum", label: "Gradient accumulation", detail: "多次 backward 前先把 loss 除以累积步数，使累积梯度对应 micro-batch 的平均值。" },
      { id: "sync", label: "DDP all-reduce", detail: "前面的 micro-step 不同步梯度；最后一次 forward/backward 触发 rank 间同步。" },
      { id: "update", label: "Clip · LR · AdamW", detail: "同步后裁剪梯度、设置 warmup/cosine 学习率并执行一次 optimizer.step。" },
      { id: "measure", label: "Val · Hella · CKPT", detail: "固定间隔切到 eval；各 rank 汇总指标，主进程记录日志并保存 checkpoint。" },
    ],
    edges: [
      { from: "shards", to: "loader", label: "tokens" },
      { from: "loader", to: "batch", label: "shift by 1" },
      { from: "batch", to: "gpt", label: "idx" },
      { from: "gpt", to: "loss", label: "logits" },
      { from: "batch", to: "loss", label: "targets" },
      { from: "loss", to: "accum", label: "backward" },
      { from: "accum", to: "sync", label: "last micro-step" },
      { from: "sync", to: "update", label: "gradients" },
      { from: "update", to: "measure", label: "interval" },
    ],
  },
  sections: [
    {
      id: "tokens-first",
      title: "先固定学习问题，再抽象 DataLoader",
      summary: "Tiny Shakespeare 先证明 next-token batch 的 shape，DataLoaderLite 随后才把切片位置变成状态。",
      paragraphs: [
        "提交顺序先加入一小段文本，编码后取 B×T+1 个 token。相邻切片分别 reshape 为 x 和 y，因此两者 shape 都是 [B,T]，但 y 的每个位置比 x 前进一个 token。交叉熵由此变成 B×T 个 next-token 分类问题。",
        "当一次手工 batch 已经跑通，DataLoaderLite 才出现。它保存 token 流、B、T 和 current_position；每次 next_batch 前进 B×T。后续换成 FineWeb shard 和 DDP 时，接口仍返回相同的 x/y，训练循环不需要重写。",
      ],
      bullets: [
        "一个 micro-batch 消费 B×T 个训练目标，但切片必须读取 B×T+1 个 token。",
        "T 决定每条样本可见的上下文长度；B 决定同一 forward 中有多少条序列。",
        "DDP 版本让不同 rank 从 B×T×rank 开始，并按 B×T×world size 跨步，避免读取同一窗口。",
      ],
      code: {
        title: "DataLoaderLite 形成错位一位的输入和目标",
        language: "python",
        snippet: `def next_batch(self):
    B, T = self.B, self.T
    buf = self.tokens[self.current_position : self.current_position+B*T+1]
    x = (buf[:-1]).view(B, T) # inputs
    y = (buf[1:]).view(B, T) # targets
    self.current_position += B * T
    if self.current_position + (B * T + 1) > len(self.tokens):
        self.current_position = 0
    return x, y`,
        sourceUrl: at(refs.loader),
        sourceLabel: "train_gpt2.py @ 631f7d6",
        note: "官方提交原样摘录；这是加入 DataLoaderLite 时的单进程版本。",
      },
      visual: {
        title: "同一 token 窗口，两个视图",
        html: `<div class="bng-shift-visual" role="img" aria-label="B 乘 T 加一的 token 窗口分别去掉末尾和开头，形成输入 x 与目标 y">
          <div class="bng-token-row"><span>buffer</span><b>42</b><b>17</b><b>9</b><b>81</b><b>6</b><b>25</b><b class="is-edge">73</b></div>
          <div class="bng-shift-brace"><i></i><span>B × T + 1</span><i></i></div>
          <div class="bng-pair-row"><div><span>x · 去掉末尾</span><code>42 17 9 81 6 25</code></div><strong>向右预测 →</strong><div><span>y · 去掉开头</span><code>17 9 81 6 25 73</code></div></div>
        </div>`,
        caption: "边界 token 只用于给最后一个输入位置提供目标；它不增加 loss 位置数。",
      },
    },
    {
      id: "model-and-weights",
      title: "模型先能前向，再用 GPT-2 权重作结构检查",
      summary: "Transformer、生成和训练损失成立后，from_pretrained 才验证参数名、shape 与矩阵方向。",
      paragraphs: [
        "可见的首个提交已包含 GPTConfig、CausalSelfAttention、MLP、Block 和 GPT 容器；后续提交依次补 forward、top-k 生成、交叉熵与小批次训练。这先确定自写模块的张量路径，再处理外部 checkpoint。",
        "GPT-2 初始化创建相同规模的本地模型，对照 Hugging Face GPT2LMHeadModel 的 state_dict。OpenAI/Hugging Face 的 Conv1D 权重与 nn.Linear 存储方向不同，因此注意力和 MLP 的四类矩阵转置后复制；其他参数必须 shape 完全相同。导入成功说明结构兼容，但训练复现仍默认从随机参数开始。",
      ],
      bullets: [
        "GPT-2 124M 配置是 12 层、12 个头、768 hidden size、1024 context。",
        "token embedding 与 lm_head 后续被 weight tying，共享同一个参数张量。",
        "加载预训练权重用于对齐实现；从零复现使用 GPT(GPTConfig(...))。",
      ],
      code: {
        title: "Conv1D 权重在复制前转置",
        language: "python",
        snippet: `transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
              'mlp.c_fc.weight', 'mlp.c_proj.weight']
assert len(sd_keys_hf) == len(sd_keys)
for k in sd_keys_hf:
    if any(k.endswith(w) for w in transposed):
        assert sd_hf[k].shape[::-1] == sd[k].shape
        with torch.no_grad():
            sd[k].copy_(sd_hf[k].t())
    else:
        assert sd_hf[k].shape == sd[k].shape
        with torch.no_grad():
            sd[k].copy_(sd_hf[k])`,
        sourceUrl: at(refs.weights),
        sourceLabel: "train_gpt2.py @ 9ac321e",
        note: "基于官方实现的排版节选；压缩了 transposed 列表，并省略 assert 的错误消息，复制分支未改写。",
      },
      visual: {
        title: "Checkpoint 对齐不是模糊匹配",
        html: `<div class="bng-weight-map" role="img" aria-label="Hugging Face GPT-2 state dict 与本地 GPT 模型逐键和逐 shape 对齐">
          <div><span>HF GPT-2</span><b>state_dict</b><small>Conv1D layout</small></div>
          <div class="is-gate"><b>key count</b><b>shape assert</b><b>transpose 4 类</b></div>
          <div><span>local GPT</span><b>state_dict</b><small>nn.Linear layout</small></div>
          <footer><i></i><span>不匹配 → assert 中止</span><i></i></footer>
        </div>`,
        caption: "显式断言让层数、命名或矩阵方向的偏差在生成结果之前暴露。",
      },
    },
    {
      id: "performance-stack",
      title: "正确性稳定后，再逐层减少 GPU 成本",
      summary: "TF32、bfloat16、torch.compile、Flash Attention 和词表对齐依次进入，每一步都保留可测的 tokens/sec。",
      paragraphs: [
        "性能提交没有同时落入一个黑盒。先允许 CUDA 的 float32 矩阵乘使用较低内部精度，再用 autocast 把适合的算子转为 bfloat16；随后 torch.compile 捕获并优化模型图；最后 scaled_dot_product_attention 取代显式构造 T×T 注意力矩阵的实现。每步之后都同步 GPU 并计算 tokens/sec。",
        "这些开关作用不同：mixed precision 改变算术 dtype，compile 改变执行计划，Flash Attention 改变注意力 kernel 与中间张量物化方式。把 vocab size 从 50,257 补到 50,304 则是 shape 对齐，让输出维度更适合硬件；它会增加未使用的 logits 槽位。",
        "最终脚本把 use_compile 设为 false，因为当时它会干扰 HellaSwag 评估和生成。这说明吞吐改动必须连同训练之外的代码路径验证。",
      ],
      bullets: [
        "bfloat16 autocast 包住 forward 和 loss；optimizer 参数仍由 PyTorch 按实际参数 dtype 管理。",
        "Flash Attention 仍执行 causal self-attention，不改变模型要学习的目标。",
        "首次 compile 包含编译开销，不能直接当稳态 step 时间。",
      ],
      code: {
        title: "Flash Attention 替代显式注意力矩阵",
        language: "python",
        snippet: `qkv = self.c_attn(x)
q, k, v = qkv.split(self.n_embd, dim=2)
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
y = y.transpose(1, 2).contiguous().view(B, T, C)
y = self.c_proj(y)`,
        sourceUrl: at(refs.flash),
        sourceLabel: "train_gpt2.py @ 7ee630c",
        note: "官方提交摘录；只移除行尾 shape 注释以缩短展示。",
      },
      visual: {
        title: "五次改动落在不同层",
        html: `<div class="bng-speed-stack" role="img" aria-label="从数学精度到执行图、注意力内核和词表 shape 的性能优化层次">
          <div style="--bng-level:1"><span>算术</span><b>TF32 matmul</b><small>float32 矩阵乘路径</small></div>
          <div style="--bng-level:2"><span>dtype</span><b>bfloat16 autocast</b><small>forward / loss</small></div>
          <div style="--bng-level:3"><span>执行图</span><b>torch.compile</b><small>graph capture</small></div>
          <div style="--bng-level:4"><span>kernel</span><b>SDPA / Flash</b><small>causal attention</small></div>
          <div style="--bng-level:5"><span>shape</span><b>50,304 vocab</b><small>对齐输出维度</small></div>
        </div>`,
        caption: "顺序用于隔离变量；同一硬件上的 tokens/sec 才能说明某一步的实际影响。",
      },
    },
    {
      id: "global-batch",
      title: "显存装不下 global batch，才需要累积与 DDP",
      summary: "gradient accumulation 在时间上拼 batch，DDP 在 rank 间拼 batch；两者共同保持每次更新的 token 数。",
      paragraphs: [
        "项目把目标 batch 写成 524,288 tokens。单个 rank 的一个 micro-step 只处理 B×T；world size 个 rank 并行处理 world size×B×T；还差的倍数由 grad_accum_steps 补足。因此 global token batch = B×T×world size×grad accumulation。",
        "每个 micro-step 的 loss 先除以 grad_accum_steps 再 backward，使本地累积对应平均 loss。DDP 若每次 backward 都 all-reduce，会重复通信；脚本只让最后一个 micro-step 同步。2024-06-17 的修复把 require_backward_grad_sync 移到 forward 之前，因为该字段也影响 DDP forward 路径。",
        "所有 rank 随后对 loss 指标求平均，但参数更新由 DDP 已同步的梯度驱动。只有 rank 0 负责日志和 checkpoint；这与所有 rank 都执行 optimizer.step 不冲突。",
      ],
      bullets: [
        "B、T、world size 或累积步数任一翻倍，global token batch 都翻倍。",
        "若要保持 global batch，增加 GPU 数时应相应减少累积步数。",
        "梯度裁剪、学习率设置和 optimizer.step 每个 global step 只执行一次。",
      ],
      code: {
        title: "只在最后一个 micro-step 同步梯度",
        language: "python",
        snippet: `for micro_step in range(grad_accum_steps):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    if ddp:
        model.require_backward_grad_sync = (
            micro_step == grad_accum_steps - 1
        )
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss = loss / grad_accum_steps
    loss_accum += loss.detach()
    loss.backward()
if ddp:
    dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)`,
        sourceUrl: at(refs.ddpFix),
        sourceLabel: "train_gpt2.py @ f6d194b",
        note: "基于官方修复提交的排版节选；只折行条件表达式，require_backward_grad_sync 仍位于 forward 之前。",
      },
      visual: {
        title: "空间并行 × 时间累积",
        html: `<div class="bng-ddp-visual" role="img" aria-label="三个 rank 分别运行四个 micro-step，最后一个 micro-step 后同步梯度并更新">
          <header><span>micro 1</span><span>micro 2</span><span>micro 3</span><span>micro 4</span><strong>update</strong></header>
          <div><b>rank 0</b><i></i><i></i><i></i><i class="is-sync"></i><em>AdamW</em></div>
          <div><b>rank 1</b><i></i><i></i><i></i><i class="is-sync"></i><em>AdamW</em></div>
          <div><b>rank 2</b><i></i><i></i><i></i><i class="is-sync"></i><em>AdamW</em></div>
          <footer><span>本地累积</span><strong>all-reduce ↑</strong><span>相同梯度</span></footer>
        </div>`,
        caption: "横向是 gradient accumulation，纵向是 DDP ranks；同步点位于最后一次 forward/backward。",
      },
    },
    {
      id: "evaluation",
      title: "训练能下降后，才加入独立的测量路径",
      summary: "验证损失、HellaSwag、生成与 checkpoint 回答不同问题，并由主进程统一记录。",
      paragraphs: [
        "切换到 FineWeb EDU 后，仓库增加 train/val shard。验证阶段 reset val loader，在 no_grad 和 autocast 中累计 20 个 batch 的 loss，再跨 rank 求平均。它衡量固定验证 token 的 next-token loss，不参与参数更新。",
        "HellaSwag 把四个候选续写渲染为 token 和 completion mask，按 mask 内每 token 的平均损失选择答案；样本按 rank 分片，最后合并正确数。生成样本则固定随机种子并做 top-k sampling，用于观察文本，不是准确率指标。",
        "checkpoint 在验证区间由主进程写入模型、配置、step 和 val_loss。源码同时注明它没有保存 optimizer state 和随机数状态，因此可保存模型快照，但不能精确恢复完整训练轨迹。",
      ],
      bullets: [
        "train loss：当前优化目标；val loss：未用于更新的 token 分布。",
        "HellaSwag：选择题 completion 指标；generation：定性样本。",
        "checkpoint：参数快照；要精确续训还需 optimizer、数据位置和 RNG state。",
      ],
      code: null,
      visual: {
        title: "同一个 checkpoint，四条读取路径",
        html: `<div class="bng-eval-fan" role="img" aria-label="训练参数分流到验证损失、HellaSwag、文本生成和 checkpoint">
          <div class="is-center"><span>raw_model</span><b>step N</b></div>
          <div><span>VAL</span><b>20 batches</b><small>mean loss</small></div>
          <div><span>HELLA</span><b>4 completions</b><small>masked avg loss</small></div>
          <div><span>SAMPLE</span><b>top-k 50</b><small>qualitative text</small></div>
          <div><span>CKPT</span><b>model · config</b><small>step · val_loss</small></div>
        </div>`,
        caption: "评估和保存都在固定间隔触发；它们不应被并入训练 loss 的 backward 路径。",
      },
    },
  ],
  lab: {
    title: "GPT-2 构建时间线与 batch / shape 实验台",
    intro: "切换里程碑查看当时新增的组件与数据流；调整 B、T、world size 和 gradient accumulation，观察 global token batch、x/y shape 与 DDP 同步位置。",
    html: `<div class="bng-lab" data-bng-lab>
      <section class="bng-timeline-panel" aria-labelledby="bng-timeline-title">
        <header><div><span>01 · COMMIT HISTORY</span><h3 id="bng-timeline-title">从模型骨架到可测训练</h3></div><p>← → 切换里程碑</p></header>
        <div class="bng-timeline" role="toolbar" aria-label="构建里程碑" data-bng-timeline>
          <button type="button" data-bng-milestone="0" aria-pressed="true"><span>01</span><b>模型</b><small>28916d9</small></button>
          <button type="button" data-bng-milestone="1" aria-pressed="false"><span>02</span><b>batch</b><small>92b5bf9</small></button>
          <button type="button" data-bng-milestone="2" aria-pressed="false"><span>03</span><b>loader</b><small>631f7d6</small></button>
          <button type="button" data-bng-milestone="3" aria-pressed="false"><span>04</span><b>权重</b><small>9ac321e</small></button>
          <button type="button" data-bng-milestone="4" aria-pressed="false"><span>05</span><b>性能</b><small>5265→7230</small></button>
          <button type="button" data-bng-milestone="5" aria-pressed="false"><span>06</span><b>优化器</b><small>105f→3a14</small></button>
          <button type="button" data-bng-milestone="6" aria-pressed="false"><span>07</span><b>累积</b><small>01be6b3</small></button>
          <button type="button" data-bng-milestone="7" aria-pressed="false"><span>08</span><b>DDP</b><small>ba2554a</small></button>
          <button type="button" data-bng-milestone="8" aria-pressed="false"><span>09</span><b>数据</b><small>69cb21f</small></button>
          <button type="button" data-bng-milestone="9" aria-pressed="false"><span>10</span><b>评估</b><small>21d3→efed</small></button>
        </div>
        <div class="bng-milestone-card" data-bng-milestone-card></div>
        <div class="bng-flow" aria-label="当前里程碑的数据流" data-bng-flow></div>
      </section>

      <section class="bng-batch-panel" aria-labelledby="bng-batch-title">
        <header><div><span>02 · GLOBAL BATCH</span><h3 id="bng-batch-title">一次参数更新由什么组成</h3></div><code>B × T × world × accumulation</code></header>
        <div class="bng-controls">
          <label><span>每卡序列数 B <output data-bng-value="batch">8</output></span><input type="range" min="1" max="64" step="1" value="8" data-bng-input="batch"></label>
          <label><span>序列长度 T <output data-bng-value="sequence">1024</output></span><input type="range" min="64" max="2048" step="64" value="1024" data-bng-input="sequence"></label>
          <label><span>world size <output data-bng-value="world">4</output></span><input type="range" min="1" max="8" step="1" value="4" data-bng-input="world"></label>
          <label><span>grad accumulation <output data-bng-value="accumulation">16</output></span><input type="range" min="1" max="32" step="1" value="16" data-bng-input="accumulation"></label>
        </div>
        <div class="bng-equation" data-bng-equation></div>
        <div class="bng-shapes" data-bng-shapes></div>
        <div class="bng-step-map" data-bng-step-map></div>
        <p class="bng-lab-note">实验只展示离散的数据组织。显存、通信时间、吞吐与收敛不会由这四个数单独决定。</p>
      </section>
      <p class="bng-live" aria-live="polite" aria-atomic="true" data-bng-live></p>
    </div>`,
  },
  takeaways: [
    "按提交顺序阅读时，每一项机制都解决前一版本已经暴露的问题。",
    "x 与 y 的一位错位定义 next-token prediction；其 shape 始终是 [B,T]。",
    "mixed precision、compile 和 Flash Attention 分别改变 dtype、执行图与 kernel。",
    "global token batch 等于 B×T×world size×gradient accumulation。",
    "DDP 的梯度同步放在最后一个 micro-step；optimizer 每个 global step 更新一次。",
    "验证、HellaSwag、生成与 checkpoint 是训练循环旁路，不是同一个指标。",
  ],
  sources: [
    { label: "build-nanogpt README", url: `${repo}/blob/master/README.md` },
    { label: "build-nanogpt 提交历史", url: `${repo}/commits/master/` },
    { label: "最终 train_gpt2.py", url: `${repo}/blob/6104ab1b53920f6e2159749676073ff7d815c1fa/train_gpt2.py` },
    { label: "FineWeb EDU 数据准备脚本", url: `${repo}/blob/6104ab1b53920f6e2159749676073ff7d815c1fa/fineweb.py` },
    { label: "HellaSwag 评估脚本", url: `${repo}/blob/${refs.eval}/hellaswag.py` },
    { label: "GPT-2 paper", url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" },
  ],
};
