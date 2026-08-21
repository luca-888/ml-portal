const upstream = "https://github.com/karpathy/llm.c/blob/f1e2ace651495b74ae22d45d1723443fd00ecd3a";

export default {
  slug: "llm-c",
  title: "llm.c",
  subtitle: "对照 C CPU reference 与 CUDA training path，拆解 GPT-2 的一次训练 step。",
  repoUrl: "https://github.com/karpathy/llm.c",
  accent: "#e85d04",
  tags: ["GPT-2", "C", "CUDA", "Training", "Memory Layout", "Mixed Precision"],
  audience: "理解 Transformer 结构，准备从张量框架继续阅读 C、CUDA 训练代码的学习者。",
  prerequisites: [
    "知道 GPT-2 block 由 LayerNorm、causal self-attention、MLP 和 residual connection 组成",
    "能阅读 C 指针、结构体和连续数组",
    "了解 CUDA grid、block、stream 与 kernel launch 的基本含义",
  ],
  outcomes: [
    "用 B、T、C、L 写出主要参数与 activation 的 shape",
    "说明 ParameterTensors 与 ActivationTensors 如何指向连续 buffer 的不同区段",
    "把 CPU forward/backward 函数与 CUDA kernel、cuBLASLt、cuDNN 调用对应起来",
    "区分 BF16 模型张量、FP32 统计量、AdamW 状态与可选 master weights",
    "沿 forward、loss、backward、optimizer 检查一次训练 step 的数据依赖",
  ],
  overview: [
    "llm.c 同时保留 `train_gpt2.c` 与 `train_gpt2.cu`。前者把 GPT-2 的 fp32 CPU forward、backward 和 AdamW 放在一个 C 文件中；后者是当前 CUDA 训练路径，调用项目内 CUDA kernels、cuBLASLt，并可选择 cuDNN attention。两条路径计算同一个训练问题，但 buffer 复用、精度和执行单元不同。",
    "本页基于官方仓库提交 f1e2ace。交互实验只做 shape、字节数和相对工作量估算，不运行模型，不预测特定 GPU 或 CPU 的实际耗时。",
  ],
  architecture: {
    title: "一次 GPT-2 训练 step 的数据路径",
    description: "参数跨 step 保留；activation 在 forward 写入，backward 逆序读取；AdamW 在梯度生成后更新参数。",
    nodes: [
      { id: "batch", label: "inputs / targets", detail: "B×T 个 token id；target 是下一 token 标签。" },
      { id: "params", label: "Parameter buffer", detail: "16 个参数张量按固定次序映射到一段连续内存。" },
      { id: "forward", label: "Forward", detail: "embedding、L 个 Transformer blocks、final LayerNorm 与词表投影。" },
      { id: "acts", label: "Activation buffer", detail: "保存 backward 所需的中间值，并包含 CUDA scratch 区。" },
      { id: "loss", label: "Cross-entropy", detail: "对 B×T 个位置计算损失，并产生 logits 梯度。" },
      { id: "backward", label: "Backward", detail: "从 classifier 向 embedding 逆序应用链式法则。" },
      { id: "grads", label: "Gradient buffer", detail: "布局与参数对应；梯度累加后交给 optimizer。" },
      { id: "adamw", label: "AdamW", detail: "读取参数、梯度、m、v；CUDA BF16 可同时维护 FP32 master weights。" },
    ],
    edges: [
      { from: "batch", to: "forward", label: "input ids" },
      { from: "params", to: "forward", label: "weights" },
      { from: "forward", to: "acts", label: "保存中间值" },
      { from: "forward", to: "loss", label: "logits" },
      { from: "batch", to: "loss", label: "targets" },
      { from: "loss", to: "backward", label: "dlogits" },
      { from: "acts", to: "backward", label: "读取" },
      { from: "params", to: "backward", label: "读取" },
      { from: "backward", to: "grads", label: "累加" },
      { from: "grads", to: "adamw", label: "更新量" },
      { from: "adamw", to: "params", label: "下一 step" },
    ],
  },
  sections: [
    {
      id: "two-paths",
      title: "1. CPU reference 与 CUDA path 的边界",
      summary: "C 文件表达计算顺序；CUDA 文件保留同一顺序，但把执行交给 kernel、cuBLASLt 或 cuDNN。",
      paragraphs: [
        "`train_gpt2.c` 使用 float 张量。forward 显式调用 LayerNorm、矩阵乘、attention、GELU 和 residual 的 CPU 函数，backward 用对应函数逆序传播。它适合核对 shape、指针偏移和梯度依赖。",
        "`train_gpt2.cu` 的 `floatX` 由编译选项选择 FP32、FP16 或 BF16。矩阵乘通过 cuBLASLt，其他算子由 `llmc/*.cuh` 中的 kernels 执行；启用 cuDNN 时 attention 改走 cuDNN frontend graph。CUDA 路径还复用 activation 区作为 scratch，避免为每个中间量保留独立数组。",
      ],
      bullets: [
        "CPU reference：函数调用与数学算子接近一一对应，activation 与 activation gradient 都保留完整布局。",
        "CUDA path：同一逻辑算子可能融合，矩阵乘由库执行，scratch buffer 会在不同阶段承载不同张量。",
        "对照阅读时先匹配输入输出 shape，再比较实现方式；函数名是否相同不是判断等价性的依据。",
      ],
      visual: {
        title: "同一 block 的两种执行方式",
        html: `<div class="lc-path-compare" role="img" aria-label="CPU reference 与 CUDA path 的算子对照">
          <div class="lc-path-compare__head"><span>逻辑算子</span><span>CPU reference</span><span>CUDA path</span></div>
          <div><b>LayerNorm</b><span>C 循环 / OpenMP</span><span>layernorm kernel</span></div>
          <div><b>QKV / projection</b><span>matmul_forward</span><span>cuBLASLt matmul</span></div>
          <div><b>Attention</b><span>C attention_forward</span><span>CUDA kernels / cuDNN</span></div>
          <div><b>Residual + LN</b><span>两个函数</span><span>fused_residual_forward5</span></div>
          <div><b>AdamW</b><span>参数循环</span><span>adamw kernel</span></div>
        </div>`,
        caption: "CUDA path 的融合改变读写次数和 launch 数量，但训练 step 的依赖顺序不变。",
      },
    },
    {
      id: "shape-layout",
      title: "2. shape 先变成长度，再变成指针偏移",
      summary: "ParameterTensors 不是 16 次独立分配；代码先计算长度，再把字段指向同一连续 allocation。",
      paragraphs: [
        "模型参数由 Vp（padding 后词表）、maxT、C 和 L 决定。每层四个主要权重矩阵合计 `12C²` 个元素：QKV 为 `3C²`，attention projection 为 `C²`，MLP 两个矩阵各为 `4C²`。embedding 与输出 projection 共享 `wte`。",
        "CPU 与 CUDA 都按 `wte → wpe → ln1w → … → lnfb` 排列参数。结构体字段只是视图；`params_memory` 才是所有权所在。activation 也使用连续 allocation，但 CPU reference 保留 23 个张量，CUDA 主路径使用 21 个张量并加入 output、scratch 等复用区。",
      ],
      bullets: [
        "`(L, B, T, C)` 的第 l 层起点是 base + `l·B·T·C`。",
        "CPU attention 保存 `(L,B,NH,T,T)` 的 preatt 与 att；因此 T 翻倍时这部分内存变为四倍。",
        "CUDA 非 cuDNN attention 仍有 T² buffer；cuDNN 路径保存的 attention stats 形状为 `(L,B,NH,T)`。",
      ],
      code: {
        title: "16 个参数张量的元素数",
        language: "c",
        snippet: `param_sizes[0] = Vp * C; // wte
param_sizes[1] = maxT * C; // wpe
param_sizes[2] = L * C; // ln1w
param_sizes[3] = L * C; // ln1b
param_sizes[4] = L * (3 * C) * C; // qkvw
param_sizes[5] = L * (3 * C); // qkvb
param_sizes[6] = L * C * C; // attprojw
param_sizes[7] = L * C; // attprojb
param_sizes[8] = L * C; // ln2w
param_sizes[9] = L * C; // ln2b
param_sizes[10] = L * (4 * C) * C; // fcw
param_sizes[11] = L * (4 * C); // fcb
param_sizes[12] = L * C * (4 * C); // fcprojw
param_sizes[13] = L * C; // fcprojb
param_sizes[14] = C; // lnfw
param_sizes[15] = C; // lnfb`,
        sourceUrl: `${upstream}/train_gpt2.c#L561-L576`,
        sourceLabel: "train_gpt2.c",
        note: "官方源码原文。Vp 是 padded_vocab_size，maxT 是 checkpoint 的最大序列长度。",
      },
      visual: {
        title: "连续参数 buffer",
        html: `<div class="lc-buffer-map" role="img" aria-label="连续参数 buffer 被划分成多个张量">
          <div class="lc-buffer-map__bar"><span data-part="embed">wte<br>Vp×C</span><span data-part="pos">wpe<br>maxT×C</span><span data-part="attn">QKV + attn</span><span data-part="mlp">MLP</span><span data-part="norm">norm / bias</span></div>
          <div class="lc-buffer-map__pointer"><span>params_memory</span><i>字段保存各分区首地址，不复制数据</i></div>
        </div>`,
        caption: "图中宽度用于说明分区，不按真实参数比例绘制；实验会按当前输入给出估算比例。",
      },
    },
    {
      id: "forward",
      title: "3. forward：从张量公式到执行单元",
      summary: "每层保持 LayerNorm → attention → residual → LayerNorm → MLP → residual 的依赖。",
      paragraphs: [
        "CPU reference 为每层计算指针偏移，然后按模型顺序调用函数。QKV 的输出 shape 是 `(B,T,3C)`；attention 输出回到 `(B,T,C)`；MLP 中间层扩张到 `(B,T,4C)`，projection 后回到 C。",
        "CUDA path 中，QKV、attention projection、MLP 两个矩阵乘都交给 cuBLASLt。LayerNorm、GELU、residual、embedding 和非 cuDNN attention 由项目 kernels 执行。当前路径把 residual 与下一次 LayerNorm 融合，也可让 cuBLASLt 融合 GELU。",
      ],
      bullets: [
        "矩阵乘主工作量随 `B·T·L·C²` 增长。",
        "非 fused attention 的分数矩阵随 `B·L·NH·T²` 增长。",
        "输出 projection 的 shape 是 `(B,T,C) × (Vp,C)ᵀ → (B,T,Vp)`，wte 在这里再次使用。",
      ],
      code: {
        title: "CPU reference 的一个 Transformer block",
        language: "c",
        snippet: `layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
residual_forward(l_residual2, residual, l_attproj, B*T*C);
layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);`,
        sourceUrl: `${upstream}/train_gpt2.c#L863-L872`,
        sourceLabel: "train_gpt2.c",
        note: "官方源码原文；外层 `for (l=0; l<L; l++)` 未包含在摘录中。",
      },
      visual: {
        title: "shape 在一个 block 中的变化",
        html: `<div class="lc-shape-flow" role="img" aria-label="GPT-2 block 张量 shape 流程">
          <span><b>residual</b>B×T×C</span><i>LN</i><span><b>QKV</b>B×T×3C</span><i>Attention</i><span><b>att out</b>B×T×C</span><i>+ residual</i><span><b>MLP</b>B×T×4C</span><i>project + add</i><span><b>next</b>B×T×C</span>
        </div>`,
        caption: "C 维在 attention 输出和 MLP projection 后恢复，因此 residual addition 的两个输入 shape 相同。",
      },
    },
    {
      id: "cuda-launch",
      title: "4. CUDA path：wrapper 决定 grid、block 与 stream",
      summary: "主训练函数组织算子，具体 wrapper 再选择 kernel 配置或库调用。",
      paragraphs: [
        "`train_gpt2.cu` 不在训练循环里直接写所有 `<<<grid, block>>>`。例如 `encoder_forward` wrapper 根据 `B·T·C` 算 grid size，并把 `cudaStream_t` 传给 launch；LayerNorm、GELU、attention 等采用相同分层方式。",
        "矩阵乘是另一条边界：`matmul_forward_cublaslt` 配置 cuBLASLt descriptors、数据类型与可能的 epilogue。启用 cuDNN 后，attention 的 forward/backward 交给 cuDNN graph；否则使用 `llmc/attention.cuh` 中的 permute、softmax 和矩阵乘组合。",
      ],
      bullets: [
        "kernel launch 的 grid/block 决定线程如何覆盖元素，但不改变逻辑 tensor shape。",
        "stream 规定同一 stream 上的顺序；训练函数在阶段边界使用同步或异步 copy。",
        "fused wrapper 同时处理 residual、LayerNorm 输出和统计量，减少中间 buffer 的往返。",
      ],
      code: {
        title: "embedding wrapper 启动 CUDA kernel",
        language: "cpp",
        snippet: `void encoder_forward(floatX* out,
                     const int* inp, const floatX* wte, const floatX* wpe,
                     int B, int T, int C, cudaStream_t stream) {
    const int N = B * T * C;
    const int block_size = 256;
    const int grid_size = ceil_div(N, block_size * x128::size);
    encoder_forward_kernel3<<<grid_size, block_size, 0, stream>>>(
        out, inp, wte, wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}`,
        sourceUrl: `${upstream}/llmc/encoder.cuh#L157-L166`,
        sourceLabel: "llmc/encoder.cuh",
        note: "官方源码原文。x128::size 表示一次 128-bit packed load/store 包含的 floatX 元素数。",
      },
      visual: {
        title: "调用栈与执行位置",
        html: `<div class="lc-launch-stack" role="img" aria-label="训练函数到 CUDA 执行单元的调用栈">
          <span data-level="host">train_gpt2.cu<br><small>shape / pointer / 顺序</small></span>
          <i>调用 wrapper</i>
          <span data-level="wrapper">llmc/*.cuh<br><small>grid / block / stream</small></span>
          <i>launch 或库调用</i>
          <div><span data-level="device">CUDA kernel</span><span data-level="library">cuBLASLt</span><span data-level="library">cuDNN</span></div>
        </div>`,
        caption: "实验切换执行路径时会标出每个训练阶段中由 kernel 或 GPU 库执行的算子。",
      },
    },
    {
      id: "backward-update",
      title: "5. loss、backward 与 update 构成闭环",
      summary: "loss 产生链式法则的起点；backward 逆序累加梯度；AdamW 改写参数供下一 step 使用。",
      paragraphs: [
        "CPU reference 先计算 softmax 与逐 token cross-entropy，再把每个位置的 loss 梯度设为 `1/(B·T)`。backward 从输出 projection 和 final LayerNorm 开始，按 `l=L-1…0` 逆序处理 MLP、attention 与 LayerNorm，最后把 embedding 梯度累加到 wte、wpe。",
        "CUDA 主路径用 `fused_classifier` 把 softmax、cross-entropy 和可选 dlogits 写回合并。gradient accumulation 时第一个 micro-step 清零梯度，后续 micro-step 使用 `+=`；只有最后一个 micro-step 完成归约并进入 optimizer。",
        "AdamW 对每个参数维护 FP32 m 与 v。CUDA BF16 路径可保留 FP32 master weights，并在更新后用 stochastic rounding 写回低精度参数；这意味着 dtype 变小不会让 optimizer 状态等比例缩小。",
      ],
      bullets: [
        "forward 保存的 mean、rstd、attention 和 MLP 中间值是 backward 的输入，不是模型参数。",
        "residual 分支在 backward 汇合，所以对应梯度需要累加，而不是覆盖。",
        "训练显存除参数外还包括 gradients、optimizer state、master weights、activations、scratch 和输入输出 buffer。",
      ],
      code: {
        title: "CPU reference 的一次训练 step",
        language: "c",
        snippet: `dataloader_next_batch(&train_loader);
gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
gpt2_zero_grad(&model);
gpt2_backward(&model);
gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);`,
        sourceUrl: `${upstream}/train_gpt2.c#L1164-L1168`,
        sourceLabel: "train_gpt2.c",
        note: "官方源码原文。CUDA 主路径把 backward、梯度归约和 gradient accumulation 合并到 `gpt2_backward_and_reduce`。",
      },
      visual: {
        title: "buffer 的读写阶段",
        html: `<div class="lc-rw-grid" role="img" aria-label="训练各阶段对 buffer 的读写关系">
          <div class="lc-rw-grid__head"><span>buffer</span><span>forward</span><span>loss</span><span>backward</span><span>optimizer</span></div>
          <div><b>parameters</b><span data-access="read">读</span><span>—</span><span data-access="read">读</span><span data-access="write">读写</span></div>
          <div><b>activations</b><span data-access="write">写</span><span data-access="read">读写</span><span data-access="read">读 / 复用</span><span>—</span></div>
          <div><b>gradients</b><span>—</span><span data-access="write">起点</span><span data-access="write">累加</span><span data-access="read">读</span></div>
          <div><b>m / v</b><span>—</span><span>—</span><span>—</span><span data-access="write">读写</span></div>
        </div>`,
        caption: "表中只表示主要依赖；CUDA 路径还会在 host/device 间复制 token、loss 和 checkpoint 数据。",
      },
    },
    {
      id: "reading-method",
      title: "6. 用 shape、所有权和执行位置阅读源码",
      summary: "每遇到一个指针，先回答 shape、owner、读写阶段和执行位置四个问题。",
      paragraphs: [
        "阅读 `train_gpt2.c` 时，可从训练循环进入 `gpt2_forward`，记录每个函数的输入输出 shape；再从 `gpt2_backward` 逆序返回。这样可先建立正确性路径，再去 CUDA wrapper 核对执行方式。",
        "阅读 `train_gpt2.cu` 时，把 `floatX*` 与 `float*` 分开标记。前者通常随编译 dtype 变化，LayerNorm 统计、loss、AdamW m/v 和 master weights 等仍使用 float。之后再检查 activation 的 allocation、recompute 条件和 scratch 别名。",
      ],
      bullets: [
        "shape：元素数由哪些维度相乘，是否包含 L、NH、T² 或 Vp。",
        "owner：指针来自 params_memory、acts_memory、grads_memory，还是临时 scratch。",
        "lifetime：值需保留到 backward，还是一个 kernel 结束后即可覆盖。",
        "executor：CPU loop、项目 CUDA kernel、cuBLASLt、cuDNN 或 NCCL。",
      ],
      visual: {
        title: "源码检查卡",
        html: `<div class="lc-check-card">
          <div><b>1 · Shape</b><span>B×T×C？L×B×NH×T²？</span></div>
          <div><b>2 · Owner</b><span>哪一段连续 allocation？</span></div>
          <div><b>3 · Lifetime</b><span>何时写入，何时可覆盖？</span></div>
          <div><b>4 · Executor</b><span>CPU、kernel 还是 GPU library？</span></div>
        </div>`,
        caption: "这四项也构成交互实验中每个 buffer 分区和时间线阶段的说明。",
      },
    },
  ],
  lab: {
    title: "训练 step 与显存布局检查器",
    intro: "调整 B、T、C、L、dtype 与执行路径，查看参数和 activation 的教学估算、主要 buffer 分区，以及 forward → loss → backward → optimizer 时间线。",
    html: `<div class="lc-lab" data-lc-lab>
      <section class="lc-lab__controls" aria-label="估算参数">
        <div class="lc-control-grid">
          <label>B · batch <input data-lc-input="b" type="number" min="1" max="64" step="1" value="4"></label>
          <label>T · sequence <input data-lc-input="t" type="number" min="16" max="2048" step="16" value="64"></label>
          <label>C · channels <input data-lc-input="c" type="number" min="64" max="4096" step="64" value="768"></label>
          <label>L · layers <input data-lc-input="l" type="number" min="1" max="96" step="1" value="12"></label>
          <label>dtype
            <select data-lc-input="dtype">
              <option value="bf16">BF16</option>
              <option value="fp32">FP32</option>
            </select>
          </label>
          <fieldset class="lc-path-toggle">
            <legend>执行路径</legend>
            <label><input type="radio" name="lc-path" value="cuda" checked> CUDA</label>
            <label><input type="radio" name="lc-path" value="cpu"> CPU reference</label>
          </fieldset>
        </div>
        <p class="lc-lab__assumption" data-lc-assumption></p>
      </section>

      <section class="lc-lab__metrics" aria-label="内存估算">
        <article><span>参数量</span><strong data-lc-metric="params">—</strong><small>由 Vp、maxT、C、L 估算</small></article>
        <article><span>参数 + 梯度</span><strong data-lc-metric="model">—</strong><small data-lc-model-note>—</small></article>
        <article><span>activation 工作区</span><strong data-lc-metric="acts">—</strong><small data-lc-acts-note>—</small></article>
        <article><span>训练内存合计</span><strong data-lc-metric="total">—</strong><small>不含 allocator 与库 workspace</small></article>
      </section>

      <div class="lc-lab__panels">
        <section class="lc-memory-panel" aria-labelledby="lc-memory-title">
          <div class="lc-panel-head"><div><span>Memory map</span><h3 id="lc-memory-title">主要 buffer 分区</h3></div><output data-lc-memory-total>—</output></div>
          <div class="lc-memory-bar" data-lc-memory-bar role="img" aria-label="训练内存分区估算"></div>
          <div class="lc-memory-list" data-lc-memory-list></div>
        </section>

        <section class="lc-shape-panel" aria-labelledby="lc-shape-title">
          <div class="lc-panel-head"><div><span>Tensor shapes</span><h3 id="lc-shape-title">当前 shape</h3></div><output data-lc-token-count>—</output></div>
          <div class="lc-shape-table" data-lc-shapes></div>
        </section>
      </div>

      <section class="lc-timeline-panel" aria-labelledby="lc-timeline-title">
        <div class="lc-panel-head"><div><span>Training step</span><h3 id="lc-timeline-title">相对工作量时间线</h3></div><output data-lc-path-label>—</output></div>
        <div class="lc-timeline" data-lc-timeline></div>
        <div class="lc-phase-detail" data-lc-phase-detail tabindex="0"></div>
      </section>

      <p class="lc-lab__live" data-lc-live aria-live="polite"></p>
    </div>`,
  },
  takeaways: [
    "CPU reference 用独立数组保留中间值；CUDA path 通过连续 allocation、scratch 复用和融合减少中间存储与 launch。",
    "参数量主要由 `12·L·C²` 和词嵌入 `Vp·C` 构成，activation 还受 B、T 以及 attention 的 T² 项影响。",
    "BF16 主要缩小参数、梯度与 activation；FP32 AdamW m/v 和可选 master weights 仍占显存。",
    "训练 step 的因果顺序是 forward → loss/dlogits → backward/gradient reduction → AdamW；GPU 优化不会改变该依赖。",
  ],
  sources: [
    { label: "llm.c repository", url: "https://github.com/karpathy/llm.c" },
    { label: "README at f1e2ace", url: `${upstream}/README.md` },
    { label: "CPU reference: train_gpt2.c", url: `${upstream}/train_gpt2.c` },
    { label: "CUDA training path: train_gpt2.cu", url: `${upstream}/train_gpt2.cu` },
    { label: "CUDA dtype definitions", url: `${upstream}/llmc/cuda_common.h#L82-L92` },
    { label: "AdamW CUDA kernels", url: `${upstream}/llmc/adamw.cuh` },
    { label: "GPT-2 paper", url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" },
  ],
};
