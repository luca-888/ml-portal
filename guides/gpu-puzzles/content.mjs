const upstream = "https://github.com/srush/GPU-Puzzles/blob/b3c4b237d7f0dc6d82055b753c8ea6e0cbb845eb";

export default {
  slug: "gpu-puzzles",
  title: "GPU Puzzles",
  subtitle: "用 Numba CUDA 练习线程索引、共享内存、同步与并行归约。",
  repoUrl: "https://github.com/srush/GPU-Puzzles",
  accent: "#16a085",
  tags: ["CUDA", "GPU Kernel", "Shared Memory", "Reduction", "Memory Coalescing"],
  audience: "会使用 Python 和 NumPy，希望理解机器学习算子如何映射到 GPU 线程的学习者。",
  prerequisites: [
    "知道数组按索引读写，并能阅读短 Python 函数",
    "理解矩阵的行、列与行主序线性地址",
    "不要求预先写过 CUDA；页面中的 warp 讨论以 NVIDIA CUDA 执行模型为背景",
  ],
  outcomes: [
    "用 blockIdx、blockDim 与 threadIdx 计算一维和二维全局坐标",
    "判断启动网格为何会产生越界线程，并放置边界检查",
    "说明 shared memory 与 syncthreads 在协作式计算中的职责",
    "画出 block 内并行归约的同步轮次",
    "比较连续与跨步地址在一个 warp 中形成的内存事务分组",
    "把线程映射迁移到逐元素算子、归约、卷积和矩阵乘法",
  ],
  overview: [
    "GPU-Puzzles 把一个 kernel 看成由许多线程分别执行的同一段函数。官方练习从单 block 的 Map、Guard 和二维映射开始，再加入多个 block、shared memory、归约、卷积与矩阵乘法。页面沿这条依赖链讲解，不提供谜题的完整填空答案。",
    "代码引用固定到官方仓库提交 b3c4b23。交互实验只模拟线程坐标和 float32 地址分组，不执行 CUDA，也不预测特定 GPU 的运行时间。",
  ],
  architecture: {
    title: "从 launch 参数到一次内存访问",
    description: "每个线程先由网格层级得到全局坐标，再经过 guard、地址映射和协作阶段；虚线含义由边标签说明。",
    nodes: [
      { id: "grid", label: "grid", detail: "由一个或多个 block 组成，覆盖问题空间。" },
      { id: "block", label: "blockIdx", detail: "标识当前 block 在 grid 中的坐标。" },
      { id: "thread", label: "threadIdx", detail: "标识线程在当前 block 中的局部坐标。" },
      { id: "global", label: "global index", detail: "blockIdx × blockDim + threadIdx。" },
      { id: "guard", label: "bounds guard", detail: "过滤网格向上取整后落在数据范围外的线程。" },
      { id: "address", label: "global address", detail: "由全局坐标和数组布局计算线性地址。" },
      { id: "shared", label: "shared memory", detail: "同一 block 内线程共享的暂存区。" },
      { id: "sync", label: "syncthreads", detail: "让 block 内线程在进入下一阶段前完成当前阶段。" },
      { id: "output", label: "output", detail: "逐元素结果、block 部分和或输出 tile。" },
    ],
    edges: [
      { from: "grid", to: "block", label: "划分" },
      { from: "block", to: "global", label: "block offset" },
      { from: "thread", to: "global", label: "local offset" },
      { from: "global", to: "guard", label: "范围判断" },
      { from: "guard", to: "address", label: "有效线程" },
      { from: "address", to: "shared", label: "分块载入" },
      { from: "shared", to: "sync", label: "阶段边界" },
      { from: "sync", to: "shared", label: "下一轮" },
      { from: "address", to: "output", label: "直接写" },
      { from: "shared", to: "output", label: "协作结果" },
    ],
  },
  sections: [
    {
      id: "execution-hierarchy",
      title: "1. grid、block 与 thread 组成坐标系",
      summary: "block 决定起点，thread 决定块内偏移，两者相加得到数据坐标。",
      paragraphs: [
        "一个 kernel launch 指定 blockspergrid 和 threadsperblock。每个线程运行同一个 call，但它看到的 blockIdx 与 threadIdx 不同。一维情况下，全局索引 i 等于 blockIdx.x × blockDim.x + threadIdx.x。",
        "如果数据长度为 10、每个 block 有 4 个线程，网格需要 3 个 block。前两个 block 覆盖 0–7，第三个 block 产生索引 8–11，其中 10 和 11 必须被 guard 过滤。blockDim 是启动配置，不会随最后一块自动缩小。",
      ],
      bullets: [
        "threadIdx 是 block 内局部坐标；不同 block 会重复出现 threadIdx.x = 0。",
        "blockIdx × blockDim 给出当前 block 覆盖区间的起点。",
        "ML 中的逐元素激活、归一化预处理和向量加法都可先从一线程一元素映射理解。",
      ],
      code: {
        title: "Puzzle 6 给出的一维全局索引",
        language: "python",
        snippet: `def map_block_test(cuda):
    def call(out, a, size) -> None:
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # FILL ME IN (roughly 2 lines)

    return call`,
        sourceUrl: `${upstream}/GPU_puzzlers.py#L225-L230`,
        sourceLabel: "GPU_puzzlers.py · Puzzle 6 - Blocks",
        note: "官方练习骨架原文；保留填空，未给出谜题答案。",
      },
      visual: {
        title: "一维索引的两部分",
        html: `<div class="gp-index-equation" role="img" aria-label="全局索引等于 block 起点加 thread 局部索引">
          <div><b>block 2</b><span>2 × blockDim 4</span><strong>起点 8</strong></div>
          <i>+</i>
          <div><b>thread 1</b><span>threadIdx.x</span><strong>偏移 1</strong></div>
          <i>=</i>
          <div data-result><b>global i</b><span>8 + 1</span><strong>索引 9</strong></div>
        </div>`,
        caption: "换一个 block 时局部线程编号不变，但 block 起点改变。",
      },
    },
    {
      id: "guards-and-2d",
      title: "2. guard 处理向上取整，二维映射处理矩阵",
      summary: "网格覆盖范围通常大于数据范围；矩阵则需要分别计算 x、y 并检查两条边界。",
      paragraphs: [
        "grid 维度通常按 ceil(size / blockDim) 计算。这样能覆盖尾部元素，也一定可能产生多余线程。访问数组前检查 i < size，是启动配置与数据长度之间的接口。二维矩阵需同时检查 x < width 与 y < height。",
        "GPU-Puzzles 的 Puzzle 4 先在单 block 中暴露 threadIdx.x 和 threadIdx.y，Puzzle 7 再扩展到二维 block grid。若矩阵采用行主序，坐标 (x, y) 的线性位置是 y × width + x；这也是判断相邻 x 线程是否读取相邻地址的基础。",
      ],
      bullets: [
        "guard 应包围读写；仅保护写入而先执行越界读取仍然不安全。",
        "二维 block 常用于图像、attention score tile 和矩阵乘法输出 tile。",
        "逻辑坐标与内存地址是两步：坐标相邻不必然表示地址相邻。",
      ],
      code: {
        title: "Puzzle 3 与 Puzzle 4 暴露的局部坐标",
        language: "python",
        snippet: `# Puzzle 3: 一维 guard 的起点
local_i = cuda.threadIdx.x
# FILL ME IN (roughly 2 lines)

# Puzzle 4: 二维线程坐标
local_i = cuda.threadIdx.x
local_j = cuda.threadIdx.y
# FILL ME IN (roughly 2 lines)`,
        sourceUrl: `${upstream}/GPU_puzzlers.py#L123-L165`,
        sourceLabel: "GPU_puzzlers.py · Puzzle 3–4",
        note: "由两个官方练习骨架节选并并列展示；未补全 guard 或写入语句。",
      },
      visual: {
        title: "5 × 3 数据与 4 × 2 blocks",
        html: `<div class="gp-guard-map" role="img" aria-label="二维网格覆盖数据后产生越界线程">
          <div class="gp-guard-map__legend"><span data-valid>有效线程</span><span data-oob>越界线程</span></div>
          <div class="gp-guard-map__grid">
            <span data-valid>0,0</span><span data-valid>1,0</span><span data-valid>2,0</span><span data-valid>3,0</span><span data-valid>4,0</span><span data-oob>5,0</span><span data-oob>6,0</span><span data-oob>7,0</span>
            <span data-valid>0,1</span><span data-valid>1,1</span><span data-valid>2,1</span><span data-valid>3,1</span><span data-valid>4,1</span><span data-oob>5,1</span><span data-oob>6,1</span><span data-oob>7,1</span>
            <span data-valid>0,2</span><span data-valid>1,2</span><span data-valid>2,2</span><span data-valid>3,2</span><span data-valid>4,2</span><span data-oob>5,2</span><span data-oob>6,2</span><span data-oob>7,2</span>
            <span data-oob>0,3</span><span data-oob>1,3</span><span data-oob>2,3</span><span data-oob>3,3</span><span data-oob>4,3</span><span data-oob>5,3</span><span data-oob>6,3</span><span data-oob>7,3</span>
          </div>
        </div>`,
        caption: "2 × 2 个 block 共启动 32 个线程，只有 15 个坐标属于矩阵。",
      },
    },
    {
      id: "shared-memory",
      title: "3. shared memory 是 block 内的协作暂存区",
      summary: "线程先把分散的全局内存载入块内暂存区，再在同步点之后复用数据。",
      paragraphs: [
        "shared memory 的作用域是一个 block。不同 block 的同名 shared 数组彼此独立。典型阶段是：每个线程加载一个或多个元素、调用 syncthreads、再读取本 block 其他线程写入的位置。",
        "syncthreads 是 block 级屏障，不是全 grid 屏障。参与同一阶段的线程必须以一致方式到达屏障；把屏障放在只由部分线程执行的分支中需要检查控制流是否会让其他线程缺席。官方 Puzzle 8 用一个演示引入 shared 数组和同步，后续 pooling、卷积与矩阵乘法才产生数据复用。",
      ],
      bullets: [
        "shared memory 减少重复 global reads，但增加加载、同步和容量管理。",
        "卷积窗口会重叠，同一输入可被邻近输出复用；矩阵乘法的 tile 也会被多次乘加。",
        "容量按 block 分配；增加 tile 尺寸可能降低一个 SM 同时驻留的 blocks 数。",
      ],
      code: {
        title: "Puzzle 8 的 shared memory 阶段",
        language: "python",
        snippet: `shared = cuda.shared.array(TPB, numba.float32)
i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
local_i = cuda.threadIdx.x

if i < size:
    shared[local_i] = a[i]
    cuda.syncthreads()

# FILL ME IN (roughly 2 lines)`,
        sourceUrl: `${upstream}/GPU_puzzlers.py#L300-L311`,
        sourceLabel: "GPU_puzzlers.py · Puzzle 8 - Shared",
        note: "官方练习骨架原文；代码展示加载和屏障，保留最后的练习填空。",
      },
      visual: {
        title: "global → shared → reuse",
        html: `<div class="gp-shared-flow" role="img" aria-label="四个线程把全局内存载入共享内存并在同步后复用">
          <div class="gp-shared-flow__row"><b>global</b><span>G8</span><span>G9</span><span>G10</span><span>G11</span></div>
          <div class="gp-shared-flow__arrows"><i>↓ T0</i><i>↓ T1</i><i>↓ T2</i><i>↓ T3</i></div>
          <div class="gp-shared-flow__row" data-shared><b>shared</b><span>S0</span><span>S1</span><span>S2</span><span>S3</span></div>
          <div class="gp-shared-flow__barrier">syncthreads()</div>
          <div class="gp-shared-flow__reuse"><span>T0 读 S0,S1</span><span>T1 读 S0,S1,S2</span><span>T2 读 S1,S2,S3</span><span>T3 读 S2,S3</span></div>
        </div>`,
        caption: "示意的是数据阶段，不对应某一题的完整答案。",
      },
    },
    {
      id: "reduction",
      title: "4. reduction 把 block 内多项压缩为部分结果",
      summary: "每轮让一部分线程合并两个局部值，同步后再缩小活动范围。",
      paragraphs: [
        "点积先让每个线程计算一对元素的乘积，再把这些局部值归约为一个和。Prefix Sum 与 Axis Sum 练习也使用 shared memory，只是输出布局不同。若数据多于一个 block，单次 kernel 可先输出每个 block 的部分和，后续再归约这些部分和。",
        "归约轮次之间存在数据依赖：下一轮读取上一轮写出的部分结果，所以需要同步。不同归约树在 shared reads、线程分歧和地址模式上不同；先验证索引覆盖和同步，再比较性能。",
      ],
      bullets: [
        "8 个局部值的二叉归约需要 3 个合并阶段。",
        "越界线程通常向 shared memory 写入加法单位元 0，而不是完全跳过会参与屏障的阶段。",
        "softmax 的 max/sum、LayerNorm 的 sum/sum-of-squares、loss 聚合都含 reduction。",
      ],
      code: {
        title: "Puzzle 12 为 block reduction 准备的状态",
        language: "python",
        snippet: `TPB = 8

def sum_test(cuda):
    def call(out, a, size: int) -> None:
        cache = cuda.shared.array(TPB, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x
        # FILL ME IN (roughly 12 lines)

    return call`,
        sourceUrl: `${upstream}/GPU_puzzlers.py#L510-L525`,
        sourceLabel: "GPU_puzzlers.py · Puzzle 12 - Prefix Sum",
        note: "官方练习骨架原文；归约步骤仍为填空，页面只解释依赖关系。",
      },
      visual: {
        title: "8 项 block reduction 的依赖轮次",
        html: `<div class="gp-reduction" role="img" aria-label="八个数经三轮归约为一个部分和">
          <div><b>载入</b><span>v0</span><span>v1</span><span>v2</span><span>v3</span><span>v4</span><span>v5</span><span>v6</span><span>v7</span></div>
          <div><b>轮 1</b><span>v0+v1</span><span>v2+v3</span><span>v4+v5</span><span>v6+v7</span></div>
          <div><b>轮 2</b><span>s0+s1</span><span>s2+s3</span></div>
          <div><b>轮 3</b><span>block sum</span></div>
        </div>`,
        caption: "每行结束后，下一行才能安全读取当前行的部分结果。",
      },
    },
    {
      id: "coalescing-to-ml",
      title: "5. 地址模式决定 warp 需要多少内存事务",
      summary: "同一 warp 的线程若访问相邻地址，请求更容易合并；跨步访问会触及更多内存段。",
      paragraphs: [
        "在 NVIDIA GPU 上，线程以 warp 为调度单位；一个 warp 含 32 个线程。对 global memory 的访问会根据线程请求覆盖的地址段合并为若干事务。对于 float32，连续 lane 访问连续元素通常比固定大步长访问触及更少的 32-byte 地址段。实际事务还受对齐、缓存和 GPU 架构影响。",
        "线程映射与内存布局需要一起设计。逐元素 kernel 通常让 lane x 对应连续元素；归约先连续载入再在 shared memory 中合并；矩阵乘法把 A/B tile 协作载入 shared memory，使每个 global load 服务多次乘加。实验用 8 个 float32 元素作为一个 32-byte 分组，展示地址覆盖数量，而不是性能预测。",
      ],
      bullets: [
        "contiguous：lane k 读取 base + k，地址分组较少。",
        "strided：lane k 读取 base + k × stride，stride 增大时通常覆盖更多分组。",
        "矩阵转置、embedding gather 和不规则稀疏访问可能天然跨步，需要借助 tiling、重排或缓存改善访问。",
      ],
      visual: {
        title: "同一批 8 个 lane 的地址覆盖",
        html: `<div class="gp-coalescing" role="img" aria-label="连续与跨步访问触及的三十二字节分组比较">
          <div><b>contiguous</b><span data-segment="0">0</span><span data-segment="0">1</span><span data-segment="0">2</span><span data-segment="0">3</span><span data-segment="0">4</span><span data-segment="0">5</span><span data-segment="0">6</span><span data-segment="0">7</span><em>1 组</em></div>
          <div><b>stride 4</b><span data-segment="0">0</span><span data-segment="0">4</span><span data-segment="1">8</span><span data-segment="1">12</span><span data-segment="2">16</span><span data-segment="2">20</span><span data-segment="3">24</span><span data-segment="3">28</span><em>4 组</em></div>
        </div>`,
        caption: "数字是 float32 元素地址；每 8 个元素对应一个教学用 32-byte 分组。",
      },
    },
  ],
  lab: {
    title: "GPU 线程映射沙盘",
    intro: "调整向量/矩阵尺寸、block dimensions、访问模式和 stride。网格显示每个 block 与 thread 的全局坐标、地址、warp 和越界状态；选定一个 block 后可逐线程播放，并比较 contiguous 与 strided 在 32-byte 分组上的覆盖。",
    html: `<div class="gp-lab" data-gp-lab tabindex="0">
      <div class="gp-lab__controls" aria-label="线程映射参数">
        <label>问题形状
          <select data-gp-shape>
            <option value="matrix">矩阵</option>
            <option value="vector">向量</option>
          </select>
        </label>
        <label>宽度 / N <input data-gp-width type="number" min="1" max="24" value="10"></label>
        <label data-gp-height-label>高度 <input data-gp-height type="number" min="1" max="12" value="5"></label>
        <label>blockDim.x <input data-gp-block-x type="number" min="1" max="16" value="4"></label>
        <label>blockDim.y <input data-gp-block-y type="number" min="1" max="8" value="2"></label>
        <label>当前访问
          <select data-gp-mode>
            <option value="contiguous">contiguous</option>
            <option value="strided">strided</option>
          </select>
        </label>
        <label>stride <input data-gp-stride type="number" min="2" max="16" value="4"></label>
        <label>播放 block <select data-gp-block></select></label>
      </div>
      <div class="gp-lab__toolbar" aria-label="播放控制">
        <button type="button" data-gp-prev>上一个线程</button>
        <button type="button" data-gp-next>下一个线程</button>
        <button type="button" data-gp-play>播放 block</button>
        <button type="button" data-gp-reset>重置步骤</button>
        <span data-gp-status>等待初始化</span>
      </div>
      <div class="gp-lab__summary" data-gp-summary></div>
      <section class="gp-lab__panel" aria-labelledby="gp-grid-title">
        <div class="gp-lab__panel-head"><h3 id="gp-grid-title">grid / block / thread</h3><span><i data-kind="active"></i>当前线程 <i data-kind="valid"></i>有效 <i data-kind="oob"></i>越界</span></div>
        <div class="gp-grid" data-gp-grid></div>
      </section>
      <div class="gp-lab__split">
        <section class="gp-lab__panel" aria-labelledby="gp-thread-title">
          <div class="gp-lab__panel-head"><h3 id="gp-thread-title">选定 block 的地址表</h3><span>warp = 32 lanes</span></div>
          <div class="gp-thread-table-wrap"><table class="gp-thread-table"><thead><tr><th>lane</th><th>thread</th><th>global</th><th>地址</th><th>32B 组</th></tr></thead><tbody data-gp-thread-table></tbody></table></div>
        </section>
        <section class="gp-lab__panel" aria-labelledby="gp-compare-title">
          <div class="gp-lab__panel-head"><h3 id="gp-compare-title">访问模式比较</h3><span>每组 8 × float32</span></div>
          <div class="gp-compare" data-gp-compare></div>
        </section>
      </div>
      <p class="gp-lab__hint">键盘：焦点不在输入控件时，←/→ 单步，空格播放或暂停。地址分组是覆盖量模型，不代表具体设备的运行周期。</p>
      <div class="gp-lab__live" data-gp-live aria-live="polite" aria-atomic="true"></div>
    </div>`,
  },
  takeaways: [
    "先写出全局坐标，再把坐标映射到数组地址；这两步不要混在一起检查。",
    "网格向上取整后必须处理越界线程，二维问题要检查两个维度。",
    "shared memory 用于 block 内复用；syncthreads 划分有依赖关系的计算阶段。",
    "reduction 是多个 ML 算子的共同子结构，通常先生成 block 部分结果。",
    "coalescing 取决于一个 warp 同时请求的地址集合，不能只看单个线程。",
    "完成沙盘后再进入官方 notebook；按 Puzzle 1→8→10→12→14 的顺序记录每题的坐标、guard、同步与访问次数。",
  ],
  sources: [
    { label: "GPU-Puzzles 官方仓库", url: "https://github.com/srush/GPU-Puzzles" },
    { label: "GPU_puzzlers.py（固定提交）", url: `${upstream}/GPU_puzzlers.py` },
    { label: "GPU_puzzlers.ipynb（固定提交）", url: `${upstream}/GPU_puzzlers.ipynb` },
    { label: "官方可视化辅助代码 lib.py", url: `${upstream}/lib.py` },
    { label: "NVIDIA CUDA C++ Programming Guide：线程层级", url: "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy" },
    { label: "NVIDIA CUDA Best Practices Guide：Coalesced Access", url: "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory" },
  ],
};
