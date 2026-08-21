const upstream = "https://github.com/linkedin/Liger-Kernel/blob/7c454f488ea522bdf408991543af922685ecebb1";

export default {
  slug: "liger-kernel",
  title: "Liger Kernel",
  subtitle: "面向大模型训练的 Triton 算子、融合损失与框架替换层。",
  repoUrl: "https://github.com/linkedin/Liger-Kernel",
  accent: "#15b8a6",
  tags: ["Triton", "Kernel Fusion", "LLM Training", "GPU Memory", "PyTorch"],
  audience: "会读 PyTorch 训练代码，想理解 Triton kernel 为什么能减少显存流量和中间张量的学习者。",
  prerequisites: [
    "知道 tensor shape、forward/backward 与 GPU 显存的基本含义",
    "能阅读短 Python 和 Triton 代码",
    "不要求写过 CUDA；页面会建立 program、tile、mask 的对应关系",
  ],
  outcomes: [
    "从 HBM 往返和 kernel launch 两个角度解释算子融合",
    "把 Triton program_id、offset 与 mask 映射到 tensor 的一行",
    "说明 RMSNorm、SwiGLU 与 Fused Linear Cross Entropy 的不同优化点",
    "计算未融合 logits 与分块 logits 的数量级差异",
    "区分示意估算、微基准和端到端训练结果",
    "知道替换 Hugging Face 模型后还需验证数值、反向与收敛",
  ],
  overview: [
    "Liger Kernel 的核心不是把 Python 写成另一种语法，而是重新安排一次训练步骤里的数据生命周期。逐个 PyTorch 算子执行时，中间结果可能写回 HBM，再被下一个 kernel 读入；融合 kernel 让同一 tile 的值停留在寄存器或片上存储中，完成更多计算后才写回。",
    "最值得教学展开的是三个尺度：RMSNorm 展示一行数据如何归约并归一化；SwiGLU 展示逐元素链如何合并并在反向中重计算；Fused Linear Cross Entropy 展示词表维度很大时，如何分块生成 logits、原地得到梯度，避免完整 BT × V 张量长期存在。",
    "页面源码链接固定到官方提交 7c454f4。交互实验是数据流与 logits 占用估算器，不运行 GPU，也不虚构速度；真实收益需要在目标硬件、shape、dtype 和训练配置上执行仓库基准。",
  ],
  architecture: {
    title: "从训练框架到一次 fused kernel",
    description: "高层 API 先选择可替换模块；autograd 连接 forward/backward；Triton program 以 tile 为单位在 HBM 与片上计算之间移动数据。",
    nodes: [
      { id: "trainer", label: "Trainer", detail: "Hugging Face、TRL、FSDP 或 DeepSpeed 驱动训练步骤。" },
      { id: "patch", label: "patch layer", detail: "按模型类型替换 RMSNorm、RoPE、MLP 或 loss 路径。" },
      { id: "module", label: "nn.Module", detail: "保持 PyTorch 接口，让调用方仍使用熟悉的模型结构。" },
      { id: "autograd", label: "autograd.Function", detail: "保存反向真正需要的状态，并把梯度交给定制 kernel。" },
      { id: "launch", label: "Triton launch", detail: "grid 决定启动多少个 program；constexpr 参数参与编译专化。" },
      { id: "tile", label: "program tile", detail: "program_id 选择一行或一个块，offsets 选择块内元素。" },
      { id: "hbm", label: "HBM", detail: "输入、权重与最终输出所在的设备显存；减少往返通常是融合的主要目标。" },
      { id: "onchip", label: "register / SRAM", detail: "tile 在片上完成 load、reduce、激活和逐元素组合。" },
      { id: "validation", label: "tests + benchmark", detail: "正确性、梯度、收敛和性能是四类不同证据。" },
    ],
    edges: [
      { from: "trainer", to: "patch", label: "启用" },
      { from: "patch", to: "module", label: "替换" },
      { from: "module", to: "autograd", label: "forward" },
      { from: "autograd", to: "launch", label: "调度" },
      { from: "launch", to: "tile", label: "grid" },
      { from: "hbm", to: "tile", label: "load" },
      { from: "tile", to: "onchip", label: "计算" },
      { from: "onchip", to: "hbm", label: "store" },
      { from: "autograd", to: "validation", label: "数值与梯度" },
      { from: "launch", to: "validation", label: "时间与显存" },
    ],
  },
  sections: [
    {
      id: "fusion-model",
      title: "1. 先数 HBM 往返，再数 FLOPs",
      summary: "短逐元素算子常受内存带宽和启动开销限制；融合的价值是缩短中间值的外部生命周期。",
      paragraphs: [
        "在 eager operator graph 中，平方、归约、缩放、激活和相乘是独立节点。若每个节点都产生设备张量，数据会在 HBM 与执行单元之间多次往返。融合后，一个 program load 进来的 tile 可连续完成多步，再 store 最终结果。",
        "这不等于任何情况下都更快。torch.compile 可能融合部分图；大矩阵乘法往往由高效库 kernel 主导；融合也可能增加寄存器压力、降低 occupancy。正确问题是：这个 shape 上减少的流量与 launch，是否超过融合带来的资源代价？",
      ],
      bullets: [
        "融合删除的是中间 materialization，不会删除数学上必须完成的运算。",
        "训练还包含 backward；只看 forward 图可能漏掉保存张量与梯度缓冲区。",
        "页面实验中的往返次数是教学示意，不替代 profiler 的实际 dram bytes。",
      ],
      visual: {
        title: "同一表达式的两种数据生命周期",
        html: `<div class="lk-fusion-strip" role="img" aria-label="未融合算子多次经过显存，融合算子在片上连续计算">
          <div><b>operator graph</b><span>HBM</span><i>load</i><strong>square</strong><i>store / load</i><strong>reduce</strong><i>store / load</i><strong>scale</strong><i>store</i></div>
          <div data-fused><b>fused kernel</b><span>HBM</span><i>load once</i><strong>square → reduce → scale</strong><i>store once</i></div>
        </div>`,
        caption: "片上停留时间变长，HBM 中间张量变少；具体执行仍由编译结果和硬件决定。",
      },
    },
    {
      id: "triton-tile",
      title: "2. Triton program 处理一个逻辑 tile",
      summary: "program_id 选择工作单元，arange 生成块内 offsets，mask 保护非整块尾部。",
      paragraphs: [
        "RMSNorm forward 把输入展平为多行。每个 Triton program 读取一行，BLOCK_SIZE 通常取不小于 hidden size 的 2 的幂；超出 n_cols 的 lanes 由 mask 禁止读写。这样公式、地址和并行映射出现在同一个 kernel 中。",
        "program 不是单个 CUDA thread。Triton 编译器会把 tile 运算映射到线程和向量指令。阅读代码时先追踪 tensor 的逻辑坐标与 mask，再讨论 warp 数、寄存器和 occupancy。",
      ],
      bullets: [
        "row_idx 决定基址；col_offsets 决定这一行内的列。",
        "mask 使 7,680 维向量可以安全放进 8,192 的逻辑 block。",
        "constexpr 让 block size 和分支在编译期专化，而不是每个元素运行时判断。",
      ],
      code: {
        title: "RMSNorm 的行映射与 masked load",
        language: "python",
        snippet: `row_idx = tl.program_id(0).to(tl.int64)
col_offsets = tl.arange(0, BLOCK_SIZE)
mask = col_offsets < n_cols

y_base = Y_ptr + row_idx * Y_row_stride
x_base = X_ptr + row_idx * X_row_stride
rstd_base = RSTD_ptr + row_idx * RSTD_row_stride

X_row = tl.load(x_base + col_offsets, mask=mask, other=0)
if elementwise_affine:
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)`,
        sourceUrl: `${upstream}/src/liger_kernel/ops/rms_norm.py#L83-L94`,
        sourceLabel: "src/liger_kernel/ops/rms_norm.py · 83–94",
        note: "官方源码节选；省略 dtype 分支和后续计算。",
      },
      visual: {
        title: "一行映射到一个 padded block",
        html: `<div class="lk-tile-map" role="img" aria-label="隐藏维度末尾由 mask 关闭多余 lanes">
          <div><b>program 17</b><span>row offset = 17 × stride</span></div>
          <div class="lk-tile-map__cells"><span>0</span><span>1</span><span>2</span><span>…</span><span>7678</span><span>7679</span><span data-mask>7680</span><span data-mask>8191</span></div>
          <p><i></i>active: load/store <i data-mask></i>masked: no memory access</p>
        </div>`,
        caption: "BLOCK_SIZE 是执行 tile 的宽度，n_cols 才是有效数据宽度。",
      },
    },
    {
      id: "rmsnorm",
      title: "3. RMSNorm 把归约与缩放留在一行内",
      summary: "同一 program 求均方、倒数平方根并缩放，反向只缓存每行一个 rstd。",
      paragraphs: [
        "RMSNorm 先对一行 x² 求和并除以 N，再计算 rsqrt(mean + eps)，最后乘权重。Liger 的 forward 在读取 X_row 后完成整条链，并只把每行一个 rstd 额外写出，供 backward 避免重复求归约。",
        "源码还区分 Llama、Gemma 与无特殊 casting 的路径。精度转换是 kernel 设计的一部分：归约常在 fp32 中完成，但输出要回到模型 dtype。性能优化不能跳过与参考实现一致的 casting 语义。",
      ],
      bullets: [
        "归约跨一行发生，因此 hidden size 直接影响 block size 与资源占用。",
        "缓存 rstd 的体积按行数增长，而不是按行数 × hidden size 增长。",
        "epsilon、offset 与权重 dtype 都会影响数值一致性测试。",
      ],
      code: {
        title: "均方、rstd 与最终缩放",
        language: "python",
        snippet: `mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
rstd = rsqrt(mean_square + eps)
tl.store(rstd_base, rstd)

X_row = X_row * rstd
if casting_mode == _CASTING_MODE_LLAMA:
    X_row = X_row.to(X_row_dtype)

if elementwise_affine:
    Y_row = X_row * (offset + W_row)
else:
    Y_row = X_row`,
        sourceUrl: `${upstream}/src/liger_kernel/ops/rms_norm.py#L118-L135`,
        sourceLabel: "src/liger_kernel/ops/rms_norm.py · 118–135",
        note: "官方源码节选；store output 紧随其后。",
      },
      visual: {
        title: "一行内的 reduce–broadcast",
        html: `<div class="lk-rms-flow" role="img" aria-label="输入行经过平方归约得到一个 rstd，再广播回各列">
          <div class="lk-rms-flow__row"><b>x row</b><span>x₀</span><span>x₁</span><span>x₂</span><span>…</span><span>xₙ</span></div>
          <div class="lk-rms-flow__reduce"><i>square + sum</i><strong>mean(x²)</strong><i>rsqrt</i><strong>rstd</strong></div>
          <div class="lk-rms-flow__row" data-output><b>y row</b><span>x₀·r·w₀</span><span>x₁·r·w₁</span><span>x₂·r·w₂</span><span>…</span><span>xₙ·r·wₙ</span></div>
        </div>`,
        caption: "归约得到一个标量，再广播到整行；rstd 是 backward 的紧凑缓存。",
      },
    },
    {
      id: "swiglu",
      title: "4. SwiGLU 融合逐元素链，反向用重计算换显存",
      summary: "SiLU(a) 与乘 b 在一次 kernel 中完成；backward 重新计算 sigmoid，不保存额外激活。",
      paragraphs: [
        "SwiGLU 的逐元素部分是 silu(a) × b。独立执行会产生 silu(a) 中间张量；Liger forward 同时加载 a、b，在片上算 sigmoid、乘法并写 c。backward 再次由 a 计算 sigmoid 与 silu，而不是从 forward 保存这些逐元素结果。",
        "重计算并非免费：它增加算术，却减少保存与读取中间激活。对于带宽受限的短逐元素链，这经常是有利交换；对于不同硬件和超宽 hidden size，仓库还存在按架构选择二维 column tiling 的路径。",
      ],
      bullets: [
        "fusion 解决 forward 的中间 materialization。",
        "recomputation 解决 backward 所需保存状态的体积。",
        "源码对 Blackwell 的宽行使用 2D grid，说明 tile 策略需要随架构调整。",
      ],
      code: {
        title: "SwiGLU forward 的完整逐元素核心",
        language: "python",
        snippet: `program_id = tl.program_id(0).to(tl.int64)
a_ptr += program_id * stride
b_ptr += program_id * stride
c_ptr += program_id * stride

col_offsets = tl.arange(0, BLOCK_SIZE)
mask = col_offsets < n_cols
a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
c_row = silu(a_row).cast(b_row.dtype) * b_row
tl.store(c_ptr + col_offsets, c_row, mask=mask)`,
        sourceUrl: `${upstream}/src/liger_kernel/ops/swiglu.py#L51-L65`,
        sourceLabel: "src/liger_kernel/ops/swiglu.py · 51–65",
        note: "官方源码节选；为控制行宽，省略了同一行中的 gate_multiplier 乘法。",
      },
      visual: {
        title: "保存激活与重计算的交换",
        html: `<div class="lk-recompute" role="img" aria-label="前向不保存 sigmoid 和 silu 中间张量，反向从 a 重算">
          <div><b>forward</b><span>a + b</span><i>sigmoid → silu → ×b</i><strong>c</strong></div>
          <div><b>saved</b><span>a + b</span><i data-cut>no sigmoid tensor</i><strong>smaller state</strong></div>
          <div><b>backward</b><span>dc + a + b</span><i>recompute sigmoid</i><strong>da + db</strong></div>
        </div>`,
        caption: "多做一小段逐元素计算，避免保存和回读同尺寸中间张量。",
      },
    },
    {
      id: "fused-linear-ce",
      title: "5. Fused Linear Cross Entropy 改变 logits 的峰值形状",
      summary: "按 token 分块执行 linear 与 cross entropy，并在 logits chunk 上原地生成梯度。",
      paragraphs: [
        "语言模型 hidden states 是 BT × H，输出权重是 V × H。普通 linear 会产生 BT × V logits；当序列和词表都大时，这个张量可能成为训练峰值。Liger 根据 V/H 计算 chunk size，只为一部分 tokens 生成 chunk × V logits。",
        "每个 chunk 进入 cross entropy kernel 后，logits buffer 被原地转换成 logits gradient，再立刻用于 grad_input 和 grad_weight。这里的“融合”跨越了模块边界和 autograd 生命周期，不只是把两个逐元素表达式放进同一个 Triton 函数。",
      ],
      bullets: [
        "完整 logits 元素数是 BT × V；分块峰值近似 chunk_size × V。",
        "chunk size 取 2 的幂，并由 ceil(V/H) 控制目标激活规模。",
        "页面估算仅计算 logits buffer；真实 peak memory 还包括权重、梯度、优化器和 allocator 行为。",
      ],
      code: {
        title: "用 V/H 决定 token chunk",
        language: "python",
        snippet: `BT, H = _input.shape
V = weight.shape[0]
BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

inc_factor = triton.cdiv(V, H)
chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))
num_chunks = triton.cdiv(BT, chunk_size)

for chunk_id in range(num_chunks):
    start_idx = chunk_id * chunk_size
    end_idx = min((chunk_id + 1) * chunk_size, BT)
    _input_chunk = _input[start_idx:end_idx]
    logits_chunk = _input_chunk @ weight.t()`,
        sourceUrl: `${upstream}/src/liger_kernel/ops/fused_linear_cross_entropy.py#L57-L114`,
        sourceLabel: "src/liger_kernel/ops/fused_linear_cross_entropy.py · 57–70, 108–114",
        note: "由同一官方函数的两个相邻逻辑片段合并展示；省略 bias 与 loss 参数。",
      },
      visual: {
        title: "完整矩形变成流式条带",
        html: `<div class="lk-logits-shape" role="img" aria-label="完整 BT 乘 V logits 与按 token 分块 logits 对比">
          <div><header><b>materialized logits</b><span>BT × V</span></header><i data-full></i><p>所有 token 的词表分数同时存在</p></div>
          <div><header><b>chunked logits</b><span>chunk × V</span></header><i data-chunk><em></em><em></em><em></em><em></em></i><p>生成 → loss / grad → 回收 → 下一块</p></div>
        </div>`,
        caption: "横向词表维度 V 仍然存在；降低的是同时驻留的 token 行数。",
      },
    },
    {
      id: "integration-validation",
      title: "6. 替换模块之后，证据链才刚开始",
      summary: "API 负责接入，单测、梯度、收敛和目标硬件基准分别回答不同问题。",
      paragraphs: [
        "apply_liger_kernel_to_llama 可以替换 RoPE、RMSNorm、SwiGLU、cross entropy 或 fused linear cross entropy。若模型已实例化，代码还会遍历现有层并修改模块；若尚未实例化，则替换建模模块中的类或 forward。启用时点会改变补丁是否覆盖已有对象。",
        "官方仓库把验证分为 transformer correctness tests、convergence tests 和 benchmark。输出接近不代表梯度正确，单步梯度正确不代表长训练收敛一致，microbenchmark 更不等于端到端吞吐。专业使用应保存 shape、dtype、GPU、版本和基线配置。",
      ],
      bullets: [
        "先做输出与梯度对照，再做短收敛测试，最后测真实训练配置。",
        "分别记录 latency、throughput、allocated memory 与 reserved memory，避免混为一个“快”。",
        "遇到性能回退时检查 shape、dtype、compile 状态、架构分支和是否真的进入 Liger 路径。",
      ],
      visual: {
        title: "从局部正确到端到端有效",
        html: `<ol class="lk-evidence-ladder">
          <li><span>01</span><b>forward parity</b><small>输出误差与特殊输入</small></li>
          <li><span>02</span><b>gradient parity</b><small>输入、权重与混合精度</small></li>
          <li><span>03</span><b>convergence</b><small>多步 loss 与权重轨迹</small></li>
          <li><span>04</span><b>kernel benchmark</b><small>固定 shape / dtype / GPU</small></li>
          <li><span>05</span><b>end-to-end</b><small>真实模型、并行与数据</small></li>
        </ol>`,
        caption: "每一层证据都不能被上一层替代；性能数字必须附带测试条件。",
      },
    },
  ],
  lab: {
    title: "Kernel Fusion Microscope",
    intro: "选择 RMSNorm、SwiGLU 或 Fused Linear CE，逐步观察 operator graph、tile mask 与 HBM 生命周期；FLCE 模式会按官方公式估算 logits 分块。",
    html: `<div class="lk-lab" data-lk-lab>
      <div class="lk-lab__controls">
        <label>观察对象<select data-lk-kernel><option value="rmsnorm">RMSNorm</option><option value="swiglu">SwiGLU</option><option value="flce">Fused Linear CE</option></select></label>
        <label>token rows <output data-lk-rows-output>4096</output><input data-lk-rows type="range" min="256" max="16384" step="256" value="4096"></label>
        <label>hidden H<select data-lk-hidden><option>1024</option><option selected>4096</option><option>8192</option><option>14336</option></select></label>
        <label data-lk-vocab-control>vocab V<select data-lk-vocab><option>32000</option><option>64000</option><option selected>128000</option><option>256000</option></select></label>
        <label>tile width<select data-lk-block><option>256</option><option>512</option><option selected>1024</option><option>2048</option></select></label>
      </div>

      <div class="lk-stagebar">
        <button type="button" data-lk-prev aria-label="上一步">←</button>
        <div data-lk-stages role="tablist" aria-label="kernel 阶段"></div>
        <button type="button" data-lk-play aria-label="播放数据流" aria-pressed="false">播放</button>
        <button type="button" data-lk-next aria-label="下一步">→</button>
      </div>

      <div class="lk-microscope" data-lk-microscope>
        <section class="lk-graph-panel">
          <header><div><span>OPERATOR GRAPH</span><h3 data-lk-title>RMSNorm</h3></div><p data-lk-stage-copy></p></header>
          <div class="lk-graphs" data-lk-graphs></div>
        </section>

        <section class="lk-device-panel">
          <header><span>PROGRAM VIEW</span><strong data-lk-program>program 0 · row 0</strong></header>
          <div class="lk-device">
            <div class="lk-hbm" data-lk-hbm><b>HBM</b><div></div></div>
            <div class="lk-transfer" aria-hidden="true"><i></i><i></i><i></i></div>
            <div class="lk-sm">
              <div class="lk-sm__head"><span>SM · program tile</span><b data-lk-util>100% active</b></div>
              <div class="lk-lanes" data-lk-lanes></div>
              <div class="lk-alu" data-lk-alu></div>
            </div>
          </div>
        </section>
      </div>

      <div class="lk-memory-panel">
        <header><div><span>MEMORY LIFETIME</span><h3 data-lk-memory-title>中间激活</h3></div><p>示意估算，不是基准结果</p></header>
        <div class="lk-memory-bars" data-lk-memory-bars></div>
        <dl class="lk-metrics" data-lk-metrics></dl>
      </div>

      <p class="lk-status" data-lk-status></p>
      <p class="sr-only" data-lk-live aria-live="polite"></p>
    </div>`,
  },
  takeaways: [
    "Liger Kernel 最统一的理解方式是数据生命周期优化：减少 HBM 往返、减少中间张量，必要时用重计算换保存。",
    "RMSNorm 适合学习 row-wise reduction，SwiGLU 适合学习逐元素 fusion，Fused Linear CE 适合学习跨模块分块和 gradient-in-forward。",
    "Triton 的 program、tile、offset 和 mask 是逻辑并行模型；它们不等同于单个 CUDA thread。",
    "logits 分块降低的是同时存在的 BT 行数，V 维仍需计算；显存降低不等于 FLOPs 消失。",
    "任何性能判断都必须附带硬件、shape、dtype、软件版本和基线；页面估算用于建立直觉，官方 benchmark 用于复现。",
  ],
  sources: [
    { label: "Liger Kernel official repository · commit 7c454f4", url: "https://github.com/linkedin/Liger-Kernel/tree/7c454f488ea522bdf408991543af922685ecebb1" },
    { label: "Liger Kernel documentation · Getting Started", url: "https://linkedin.github.io/Liger-Kernel/Getting-Started/" },
    { label: "Liger Kernel documentation · Contributing and repository structure", url: "https://linkedin.github.io/Liger-Kernel/contributing/" },
    { label: "Liger Kernel technical report · arXiv:2410.10989", url: "https://arxiv.org/abs/2410.10989" },
    { label: "Official RMSNorm implementation", url: `${upstream}/src/liger_kernel/ops/rms_norm.py` },
    { label: "Official SwiGLU implementation", url: `${upstream}/src/liger_kernel/ops/swiglu.py` },
    { label: "Official Fused Linear Cross Entropy implementation", url: `${upstream}/src/liger_kernel/ops/fused_linear_cross_entropy.py` },
    { label: "Official benchmark scripts", url: "https://github.com/linkedin/Liger-Kernel/tree/7c454f488ea522bdf408991543af922685ecebb1/benchmark/scripts" },
  ],
};
