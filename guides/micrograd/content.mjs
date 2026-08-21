const upstream = "https://github.com/karpathy/micrograd/blob/7bc720e951fe422b8f8814aa5aa1b64121d26b4c";

export default {
  slug: "micrograd",
  title: "micrograd",
  subtitle: "沿标量计算图理解反向模式自动微分和多层感知机训练。",
  repoUrl: "https://github.com/karpathy/micrograd",
  accent: "#e76f51",
  tags: ["Autograd", "Backpropagation", "MLP", "SGD"],
  audience: "会读 Python，想从计算图层面理解梯度如何产生和传播的学习者。",
  prerequisites: [
    "知道导数表示输出对输入的局部变化率",
    "了解神经元由加权和与非线性函数组成",
    "能阅读 Python 类、闭包和运算符重载",
  ],
  outcomes: [
    "说明 Value 如何在前向计算时同时构造有向无环图",
    "用局部导数和链式法则计算每个节点的梯度",
    "解释为什么 backward 必须使用反向拓扑顺序并累加梯度",
    "从 Neuron、Layer、MLP 追踪到清零梯度和一次 SGD 更新",
  ],
  overview: [
    "micrograd 的 Value 只保存一个标量。Python 运算符生成新的 Value，并把输入节点、运算符和一个反向闭包记录下来。神经网络中的矩阵运算因此被展开为标量加法、乘法、幂和 ReLU 组成的有向无环图。",
    "页面基于官方仓库 master 的提交 7bc720e。交互实验实现同一组标量规则，用于观察计算图、局部导数、反向拓扑顺序、梯度累加和有限差分；它不导入 micrograd，也不执行 Python。",
  ],
  architecture: {
    title: "从标量输入到参数更新",
    description: "上半段是前向构图，下半段是反向传播与优化；每条反向边都把上游梯度乘以局部导数。",
    nodes: [
      { id: "inputs", label: "Value inputs", detail: "输入 x、参数 w 和偏置 b 各自保存 data 与 grad。" },
      { id: "ops", label: "Scalar ops", detail: "加法、乘法、幂和 ReLU 产生新 Value。" },
      { id: "graph", label: "Forward DAG", detail: "_prev 记录父节点，_op 记录生成节点的运算。" },
      { id: "loss", label: "Loss", detail: "最终标量作为反向传播起点，梯度种子为 1。" },
      { id: "topo", label: "Topological order", detail: "深度优先后序保证父节点在其所有使用者之后反传。" },
      { id: "closures", label: "_backward closures", detail: "每种运算闭包计算输入的局部梯度贡献。" },
      { id: "grads", label: "Accumulated grads", detail: "同一 Value 经多条路径影响 loss 时，贡献用 += 汇合。" },
      { id: "update", label: "SGD update", detail: "参数沿负梯度方向移动，然后重新构建下一轮图。" },
    ],
    edges: [
      { from: "inputs", to: "ops", label: "参与运算" },
      { from: "ops", to: "graph", label: "记录 _prev / _op" },
      { from: "graph", to: "loss", label: "前向值" },
      { from: "loss", to: "topo", label: "grad = 1" },
      { from: "topo", to: "closures", label: "逆序调用" },
      { from: "closures", to: "grads", label: "链式法则" },
      { from: "grads", to: "update", label: "p.grad" },
      { from: "update", to: "inputs", label: "下一次前向" },
    ],
  },
  sections: [
    {
      id: "scalar-value",
      title: "1. Value 在计算时构图",
      summary: "每次标量运算既算出 data，也把结果与输入节点连接起来。",
      paragraphs: [
        "Value.data 是前向值，Value.grad 初始为 0。_prev 保存直接输入，_op 只用于标记运算，_backward 初始为空函数。图没有单独的构建阶段：执行 a + b 时，结果节点已经持有到 a、b 的连接。",
        "加法结果 out 的局部规则是 ∂out/∂a = 1、∂out/∂b = 1。闭包捕获 self、other 和 out，等到最终 loss 调用 backward 时才读取 out.grad 并写入输入梯度。",
      ],
      bullets: [
        "叶子节点通常是数据或参数；中间节点是运算结果。",
        "图是动态的：控制流实际执行了哪些运算，本轮就生成哪些节点。",
        "Value 是标量，因此一个神经元会展开为多个乘法和加法节点。",
      ],
      code: {
        title: "Value 元数据与加法节点",
        language: "python",
        snippet: `def __init__(self, data, _children=(), _op=''):
    self.data = data
    self.grad = 0
    # internal variables used for autograd graph construction
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op # the op that produced this node, for graphviz / debugging / etc

def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
        self.grad += out.grad
        other.grad += out.grad
    out._backward = _backward

    return out`,
        sourceUrl: `${upstream}/micrograd/engine.py#L5-L22`,
        sourceLabel: "micrograd/engine.py",
        note: "官方源码原文；截取构造函数与加法实现。",
      },
      visual: {
        title: "一次前向运算生成三类信息",
        html: `<div class="mg-forward-cards" role="img" aria-label="Value 前向运算同时生成数值、图连接和反向规则">
          <div><b>数值</b><code>out.data = a.data + b.data</code></div>
          <div><b>连接</b><code>out._prev = {a, b}</code></div>
          <div><b>规则</b><code>out._backward = closure</code></div>
        </div>`,
        caption: "data 立即可用；图连接与闭包留到 backward 使用。",
      },
    },
    {
      id: "local-chain-rule",
      title: "2. 局部导数通过链式法则接到 loss",
      summary: "节点只需知道自身运算的局部导数，不需要知道整张图。",
      paragraphs: [
        "设 m = xw，y = m + b。乘法边的局部导数是 ∂m/∂x = w、∂m/∂w = x；加法边的局部导数都是 1。若 y 就是最终输出，则 ∂y/∂m = 1，所以 ∂y/∂x = (∂y/∂m)(∂m/∂x) = w。",
        "直观上，上游梯度表示“最终输出对当前节点有多敏感”，局部导数表示“当前节点对某个输入有多敏感”。两者相乘，得到最终输出对该输入的敏感度。ReLU 的局部导数在输出大于 0 时为 1，否则为 0。",
      ],
      bullets: [
        "加法：梯度原样分发到两个输入。",
        "乘法：给一个输入的梯度要乘另一个输入的前向值。",
        "幂：xⁿ 的局部导数是 n·xⁿ⁻¹。",
        "ReLU：正区间通过梯度，非正区间阻断梯度。",
      ],
      visual: {
        title: "边上的局部导数",
        html: `<div class="mg-rule-strip" role="img" aria-label="加法乘法幂和 ReLU 的局部导数">
          <span><b>a + b</b><i>∂/∂a = 1</i></span>
          <span><b>a × b</b><i>∂/∂a = b</i></span>
          <span><b>aⁿ</b><i>∂/∂a = n·aⁿ⁻¹</i></span>
          <span><b>ReLU(a)</b><i>a &gt; 0 ? 1 : 0</i></span>
        </div>`,
        caption: "实验会把这些局部导数直接写在计算图的边旁。",
      },
    },
    {
      id: "topological-backward",
      title: "3. 反向传播需要反向拓扑顺序",
      summary: "先收集后序拓扑列表，再从 loss 向叶子节点依次调用闭包。",
      paragraphs: [
        "build_topo 用深度优先搜索先访问 _prev，再把当前节点加入 topo。这样列表中的父节点总在结果节点之前。reversed(topo) 则保证一个节点的所有下游使用者先完成贡献，再轮到它继续向父节点传播。",
        "self.grad = 1 是导数种子，因为 ∂loss/∂loss = 1。随后每个 _backward 只完成一跳：把当前节点的 grad 乘局部导数，并累加到直接输入。",
      ],
      bullets: [
        "visited 防止共享子图被重复加入拓扑列表。",
        "顺序错误会在节点尚未收齐下游贡献时过早向前传播。",
        "backward 不会自动清零旧梯度；多轮训练前应调用 zero_grad。",
      ],
      code: {
        title: "建立拓扑序并逆序执行闭包",
        language: "python",
        snippet: `def backward(self):

    # topological order all of the children in the graph
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)

    # go one variable at a time and apply the chain rule to get its gradient
    self.grad = 1
    for v in reversed(topo):
        v._backward()`,
        sourceUrl: `${upstream}/micrograd/engine.py#L54-L70`,
        sourceLabel: "micrograd/engine.py",
        note: "官方源码原文。",
      },
      visual: {
        title: "前向顺序与反向顺序",
        html: `<div class="mg-two-pass" role="img" aria-label="前向计算由叶到根，反向传播由根到叶">
          <div><b>forward</b><span>x,w,b</span><i>→</i><span>×</span><i>→</i><span>+</span><i>→</i><span>y</span></div>
          <div><b>backward</b><span>∂y/∂y=1</span><i>→</i><span>+</span><i>→</i><span>×</span><i>→</i><span>∂y/∂x,w,b</span></div>
        </div>`,
        caption: "反向播放器每一步对应 reversed(topo) 中的一个节点。",
      },
    },
    {
      id: "gradient-accumulation",
      title: "4. 分叉路径的梯度必须相加",
      summary: "同一 Value 多次参与计算时，loss 对它的导数是各条路径贡献之和。",
      paragraphs: [
        "在 y = xw + x² + b 中，x 同时经过乘法 xw 和平方 x² 到达 y。总导数是 ∂y/∂x = w + 2x。两个中间节点各自只产生一项，最终在 x.grad 上汇合。",
        "这也是源码使用 += 而不是 = 的原因。若赋值覆盖，后执行的路径会抹掉先到达的贡献。参数共享、残差连接和一个参数在批内多次使用都会产生这种汇合。",
      ],
      bullets: [
        "链式法则负责一条路径上的乘法。",
        "多变量链式法则还要求对所有路径求和。",
        "实验选择“共享 x”表达式后，可逐步看到 x.grad 两次增长。",
      ],
      visual: {
        title: "共享节点的两条梯度路径",
        html: `<div class="mg-diamond" role="img" aria-label="x 经乘法和平方两条路径影响输出">
          <span class="mg-diamond__source">x</span>
          <span>x × w<br><small>贡献 w</small></span>
          <span>x²<br><small>贡献 2x</small></span>
          <span class="mg-diamond__sink">y<br><small>合计 w + 2x</small></span>
        </div>`,
        caption: "路径汇合处对应梯度累加，而不是选择其中一条路径。",
      },
    },
    {
      id: "neuron-layer-mlp",
      title: "5. 神经网络是 Value 运算的容器",
      summary: "Neuron 组合加权和与 ReLU，Layer 组合多个 Neuron，MLP 再串联多个 Layer。",
      paragraphs: [
        "Neuron.__call__ 计算 b + Σwᵢxᵢ，并按 nonlin 决定是否调用 ReLU。Layer 对同一输入调用多个 Neuron；MLP 把上一层输出交给下一层，并让最后一层使用线性输出。",
        "这些类没有实现新的自动微分规则。权重和偏置本身就是 Value，因此前向调用会自然生成标量图。parameters 递归展平所有参数，供 zero_grad 和优化器更新遍历。",
      ],
      bullets: [
        "Neuron.parameters 返回权重列表和偏置。",
        "Layer.parameters 展平所有神经元参数。",
        "MLP.parameters 再展平所有层参数。",
      ],
      code: {
        title: "MLP 串联 Layer 并汇总参数",
        language: "python",
        snippet: `class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]`,
        sourceUrl: `${upstream}/micrograd/nn.py#L45-L57`,
        sourceLabel: "micrograd/nn.py",
        note: "官方源码原文；省略 __repr__。",
      },
      visual: {
        title: "组合层级与实际计算图",
        html: `<div class="mg-container-stack" role="img" aria-label="MLP Layer Neuron 和 Value 运算的层级">
          <div><b>MLP</b><span>Layer 0</span><span>Layer 1</span><span>Layer 2</span></div>
          <div><b>Layer</b><span>Neuron 0</span><span>Neuron 1</span><span>…</span></div>
          <div><b>Neuron</b><span>w₀x₀</span><span>+ w₁x₁</span><span>+ b</span><span>→ ReLU</span></div>
          <div><b>Autograd</b><span>Value</span><span>×</span><span>+</span><span>ReLU</span></div>
        </div>`,
        caption: "类层级用于组织参数；梯度仍在最底层 Value 图上传播。",
      },
    },
    {
      id: "training-step",
      title: "6. 一次训练更新会丢弃旧图并重新前向",
      summary: "训练循环依次执行 forward、zero_grad、backward 和参数原地更新。",
      paragraphs: [
        "官方 demo 先计算当前 batch 的 loss，再清零所有参数梯度并从 total_loss 反传。SGD 更新 p.data ← p.data − ηp.grad；负号表示沿 loss 下降方向移动。",
        "下一轮调用 model 会用更新后的 data 创建一张新图。micrograd 没有持久的静态图或优化器状态；示例学习率从 1.0 线性下降到约 0.109。若省略 zero_grad，本轮梯度会继续加到上一轮梯度上。",
      ],
      bullets: [
        "forward：得到预测、损失和本轮计算图。",
        "zero_grad：清除参数上一次反向传播留下的梯度。",
        "backward：按逆拓扑序填充参数梯度。",
        "update：修改参数 data，不修改已完成的旧图。",
      ],
      code: {
        title: "demo 中的反向传播与 SGD 更新",
        language: "python",
        snippet: `# backward
model.zero_grad()
total_loss.backward()

# update (sgd)
learning_rate = 1.0 - 0.9*k/100
for p in model.parameters():
    p.data -= learning_rate * p.grad

if k % 1 == 0:
    print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")`,
        sourceUrl: `${upstream}/demo.ipynb`,
        sourceLabel: "demo.ipynb（optimization 单元）",
        note: "官方 notebook 源码原文；截取 backward、update 与日志部分。",
      },
      visual: {
        title: "训练步的状态变化",
        html: `<div class="mg-train-cycle" role="img" aria-label="前向清零反向更新组成训练循环">
          <span><b>1</b>forward<small>新建 DAG</small></span>
          <span><b>2</b>zero_grad<small>参数 grad = 0</small></span>
          <span><b>3</b>backward<small>累加梯度</small></span>
          <span><b>4</b>update<small>p.data -= ηp.grad</small></span>
        </div>`,
        caption: "修改参数后再次 forward，才能得到新参数对应的 loss。",
      },
    },
  ],
  lab: {
    title: "可编辑计算图与反向传播播放器",
    intro: "选择表达式并修改 x、w、b。图中节点显示 value/grad，边显示局部导数；“下一步”按反向拓扑顺序执行一个闭包。有限差分表用数值扰动核对 x、w、b 的最终解析梯度。键盘：→ 下一步，空格播放/暂停，R 重置。",
    html: `<div class="mg-lab" data-mg-lab tabindex="0">
      <section class="mg-lab__controls" aria-label="计算图参数">
        <label>表达式
          <select data-mg-expression>
            <option value="linear">y = x × w + b</option>
            <option value="relu">y = ReLU(x × w + b)</option>
            <option value="shared">y = x × w + x² + b</option>
          </select>
        </label>
        <label>x <input type="number" step="0.1" min="-10" max="10" value="2" data-mg-input="x"></label>
        <label>w <input type="number" step="0.1" min="-10" max="10" value="-3" data-mg-input="w"></label>
        <label>b <input type="number" step="0.1" min="-10" max="10" value="1" data-mg-input="b"></label>
        <div class="mg-lab__buttons">
          <button type="button" data-mg-reset>重建图</button>
          <button type="button" data-mg-step>下一步</button>
          <button type="button" data-mg-play>连续播放</button>
        </div>
      </section>

      <section class="mg-lab__status" aria-label="反向传播状态">
        <div><span>阶段</span><strong data-mg-phase>等待反向</strong></div>
        <div><span>当前节点</span><strong data-mg-current>—</strong></div>
        <div><span>反向序号</span><strong data-mg-progress>0 / 0</strong></div>
        <div><span>输出 y</span><strong data-mg-output>0</strong></div>
      </section>

      <section class="mg-lab__graph-panel" aria-labelledby="mg-graph-title">
        <div class="mg-panel-title"><h4 id="mg-graph-title">计算图</h4><span>实线：前向依赖 · 箭头方向：反向梯度</span></div>
        <div class="mg-lab__graph" data-mg-graph></div>
        <div class="mg-lab__legend"><span data-kind="leaf">叶子 / 参数</span><span data-kind="op">运算结果</span><span data-kind="active">当前闭包</span><span data-kind="done">已执行</span></div>
      </section>

      <div class="mg-lab__bottom">
        <section class="mg-lab__topology" aria-labelledby="mg-topo-title">
          <div class="mg-panel-title"><h4 id="mg-topo-title">反向拓扑队列</h4><span>从左到右执行</span></div>
          <ol data-mg-topology></ol>
          <div class="mg-formula" data-mg-formula>点击“下一步”给输出梯度置 1。</div>
        </section>

        <section class="mg-lab__check" aria-labelledby="mg-check-title">
          <div class="mg-panel-title"><h4 id="mg-check-title">有限差分核对</h4><span>ε = 10⁻⁴</span></div>
          <table>
            <thead><tr><th>变量</th><th>解析 grad</th><th>数值 grad</th><th>差值</th></tr></thead>
            <tbody data-mg-check></tbody>
          </table>
          <p>数值导数使用 <code>[f(v+ε)-f(v-ε)]/(2ε)</code>。它通过两次前向扰动估计斜率，不读取计算图。</p>
        </section>
      </div>

      <div class="mg-lab__live" data-mg-live aria-live="polite" aria-atomic="true"></div>
    </div>`,
  },
  takeaways: [
    "自动微分不是符号化简：它记录实际执行的运算，并在图上复用局部导数。",
    "反向模式从一个标量输出出发，一次得到该输出对全部叶子和参数的梯度。",
    "反向拓扑顺序保证节点收齐下游贡献后再传播；+= 保证多条路径的贡献不会覆盖。",
    "Neuron、Layer 和 MLP 只组织 Value 运算与参数；训练循环负责清零、反传和修改 data。",
  ],
  sources: [
    { label: "micrograd 官方仓库", url: "https://github.com/karpathy/micrograd" },
    { label: "Value 自动微分引擎", url: `${upstream}/micrograd/engine.py` },
    { label: "Neuron、Layer 与 MLP", url: `${upstream}/micrograd/nn.py` },
    { label: "MLP 训练示例", url: `${upstream}/demo.ipynb` },
    { label: "官方计算图可视化 notebook", url: `${upstream}/trace_graph.ipynb` },
  ],
};
