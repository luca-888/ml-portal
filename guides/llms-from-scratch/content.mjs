const upstream = "https://github.com/rasbt/LLMs-from-scratch/blob/ec0bac5e1d854306a8ea6c308da49aae22479bd1";

export default {
  slug: "llms-from-scratch",
  title: "LLMs from Scratch",
  subtitle: "沿 token ids、张量形状和一次 forward pass 理解 GPT 语言模型。",
  repoUrl: "https://github.com/rasbt/LLMs-from-scratch",
  accent: "#2563eb",
  tags: ["Tokenizer", "Attention", "GPT", "Pretraining", "Instruction Tuning"],
  audience: "会写 Python、了解 PyTorch 张量，希望从实现层面理解 GPT 训练和指令微调的读者。",
  prerequisites: [
    "理解矩阵乘法、softmax 和交叉熵",
    "知道 batch、序列长度和特征维度的含义",
    "能阅读 nn.Module、Dataset 与训练循环",
  ],
  outcomes: [
    "从文本构造右移一位的输入与目标 token",
    "写出多头注意力各阶段的张量形状",
    "沿 GPTModel.forward 从 token ids 追踪到 logits",
    "区分预训练、指令微调和生成质量评估的输入输出",
  ],
  overview: [
    "该仓库按章节实现 tokenizer 数据管线、注意力、GPT 模型、预训练、分类微调与指令微调。理解它的主线不是记住类名，而是持续追踪三个轴：batch B、序列长度 T、嵌入维度 D。",
    "本页基于官方仓库提交 ec0bac5。交互实验使用确定性的离线小矩阵解释形状和因果遮罩，不加载仓库权重，也不复现书中的训练结果。",
  ],
  architecture: {
    title: "从文本到可生成回复的路径",
    description: "训练沿上方路径把下一 token 误差反向传播；推理把最后一个位置的 logits 采样成新 token，再送回模型。",
    nodes: [
      { id: "text", label: "Text", detail: "原始文本或格式化后的 instruction、input、response。" },
      { id: "tokens", label: "Token IDs", detail: "Tokenizer 把文本映射为整数序列，并构造右移目标。" },
      { id: "embed", label: "Embeddings", detail: "token embedding 与 position embedding 相加，形状为 [B,T,D]。" },
      { id: "attn", label: "Causal Attention", detail: "Q/K/V 分头计算，遮住当前位置右侧的 key。" },
      { id: "blocks", label: "Transformer Blocks", detail: "注意力、前馈网络、LayerNorm 与残差连接重复堆叠。" },
      { id: "logits", label: "Logits", detail: "输出头把每个位置的 D 维状态映射到词表 V。" },
      { id: "loss", label: "Next-token Loss", detail: "训练时用 logits 与右移目标计算交叉熵。" },
      { id: "reply", label: "Generated Reply", detail: "推理时选择下一个 token，循环追加到上下文。" },
    ],
    edges: [
      { from: "text", to: "tokens", label: "encode" },
      { from: "tokens", to: "embed", label: "lookup" },
      { from: "embed", to: "attn", label: "Q/K/V" },
      { from: "attn", to: "blocks", label: "residual" },
      { from: "blocks", to: "logits", label: "output head" },
      { from: "logits", to: "loss", label: "training" },
      { from: "logits", to: "reply", label: "sampling" },
      { from: "reply", to: "tokens", label: "append" },
    ],
  },
  sections: [
    {
      id: "tokens-and-targets",
      title: "1. Tokenizer 决定模型看到的离散序列",
      summary: "语言模型不直接预测字符串，而是在整数词表上做下一 token 分类。",
      paragraphs: [
        "Tokenizer.encode 把文本变成 token ids。训练数据再用滑动窗口切出固定长度 T：输入取位置 0…T−1，目标取位置 1…T，因此每个输入位置都监督它右侧的 token。stride 决定相邻样本重叠多少，不改变单个样本的形状。",
        "设 batch size 为 B，则输入和目标都是 [B,T]。token id 本身没有连续数值含义；nn.Embedding 用它索引一行可学习向量，才得到 [B,T,D]。",
      ],
      bullets: [
        "词表大小 V 决定 embedding 表和输出 logits 的最后一维。",
        "上下文长度限制一次 forward 可接收的最大 T。",
        "目标右移一位，所以位置 t 的 loss 对应 token t+1。",
      ],
      code: {
        title: "滑动窗口构造输入与下一 token 目标",
        language: "python",
        snippet: `for i in range(0, len(token_ids) - max_length, stride):
    input_chunk = token_ids[i:i + max_length]
    target_chunk = token_ids[i + 1: i + max_length + 1]
    self.input_ids.append(torch.tensor(input_chunk))
    self.target_ids.append(torch.tensor(target_chunk))`,
        sourceUrl: `${upstream}/ch04/01_main-chapter-code/gpt.py#L24-L28`,
        sourceLabel: "ch04/01_main-chapter-code/gpt.py",
        note: "官方源码原文；该文件汇总第 2–4 章代码。",
      },
      visual: {
        title: "同一序列错开一格",
        html: `<div class="lfs-shift" role="img" aria-label="输入 token 与目标 token 右移一位">
          <div><b>input [B,T]</b><span>机器</span><span>学习</span><span>从</span><span>数据</span></div>
          <div><b>target [B,T]</b><i></i><span>学习</span><span>从</span><span>数据</span><span>开始</span></div>
          <svg viewBox="0 0 640 48" aria-hidden="true"><path d="M142 8 C142 36 238 12 238 39 M238 8 C238 36 334 12 334 39 M334 8 C334 36 430 12 430 39 M430 8 C430 36 526 12 526 39"/></svg>
        </div>`,
        caption: "四个位置同时产生四个监督信号；不是只监督序列末尾。",
      },
    },
    {
      id: "embedding-shapes",
      title: "2. Embedding 把两个离散索引系相加",
      summary: "token 表示内容索引，position 表示顺序索引，相加后保持 [B,T,D]。",
      paragraphs: [
        "token embedding 表的形状是 [V,D]，输入 [B,T] 查表后得到 [B,T,D]。position embedding 表的形状是 [context_length,D]，当前序列取前 T 行得到 [T,D]；广播到 batch 后与 token embedding 相加。",
        "相加要求两者最后一维同为 D。这里不是把位置特征拼到 token 特征上，因此 Transformer block 始终接收固定宽度 D。",
      ],
      bullets: [
        "B 只表示并行样本数，不参与参数矩阵尺寸。",
        "T 控制注意力矩阵的两条边，计算和显存随 T² 增长。",
        "D 必须能被注意力头数 H 整除，每头宽度 dₕ=D/H。",
      ],
      visual: {
        title: "查表、广播与相加",
        html: `<div class="lfs-embedding" role="img" aria-label="token embedding 和 position embedding 相加">
          <div class="lfs-tensor"><small>token ids</small><strong>[B,T]</strong><span>lookup V×D</span></div>
          <i>→</i><div class="lfs-stack"><span>token [B,T,D]</span><b>＋</b><span>position [1,T,D]</span></div>
          <i>→</i><div class="lfs-tensor lfs-tensor--accent"><small>block input</small><strong>[B,T,D]</strong><span>固定宽度 D</span></div>
        </div>`,
        caption: "position embedding 在 batch 轴广播；不会新增一个特征维度。",
      },
    },
    {
      id: "causal-attention",
      title: "3. Causal self-attention 同时改变排列和信息范围",
      summary: "Q/K/V 先从 [B,T,D] 分成 H 个头，再生成每头 [T,T] 的注意力矩阵。",
      paragraphs: [
        "三个线性层把 x 投影为 Q、K、V，形状仍为 [B,T,D]。view 增加头轴得到 [B,T,H,dₕ]，transpose 后变为 [B,H,T,dₕ]，便于逐头执行 QKᵀ。结果 attention scores 为 [B,H,T,T]。",
        "因果遮罩把每一行右上方的未来位置设为负无穷；除以 √dₕ 后做 softmax。这样第 t 行概率只分配给 0…t，训练时所有位置可以并行计算，同时仍遵守自回归依赖。",
      ],
      bullets: [
        "Query 行表示当前要更新的位置，Key 列表示它可以读取的位置。",
        "多头并不改变总宽度 D，只把 D 划分为 H 份并行子空间。",
        "关闭 causal mask 可用于观察差异，但会让预训练位置看到其监督目标及更后的 token。",
      ],
      code: {
        title: "分头并计算被遮罩的 scaled dot-product attention",
        language: "python",
        snippet: `keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
values = values.view(b, num_tokens, self.num_heads, self.head_dim)
queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

keys = keys.transpose(1, 2)
queries = queries.transpose(1, 2)
values = values.transpose(1, 2)

attn_scores = queries @ keys.transpose(2, 3)
mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
attn_scores.masked_fill_(mask_bool, -torch.inf)
attn_weights = torch.softmax(
    attn_scores / keys.shape[-1]**0.5, dim=-1
)`,
        sourceUrl: `${upstream}/ch04/01_main-chapter-code/gpt.py#L80-L101`,
        sourceLabel: "ch04/01_main-chapter-code/gpt.py",
        note: "官方源码原文；省略投影和 dropout，只保留形状变化、score、mask 与 softmax。",
      },
      visual: {
        title: "第 0 个 head 的可见区域",
        html: `<div class="lfs-mask" role="img" aria-label="四乘四因果注意力矩阵，下三角可见">
          <div class="lfs-mask__corner">Q↓ K→</div><b>0</b><b>1</b><b>2</b><b>3</b>
          <b>0</b><span>可见</span><i>mask</i><i>mask</i><i>mask</i>
          <b>1</b><span>可见</span><span>可见</span><i>mask</i><i>mask</i>
          <b>2</b><span>可见</span><span>可见</span><span>可见</span><i>mask</i>
          <b>3</b><span>可见</span><span>可见</span><span>可见</span><span>可见</span>
        </div>`,
        caption: "每一行 softmax 后总和为 1；被遮罩单元的概率为 0。",
      },
    },
    {
      id: "forward-pass",
      title: "4. Transformer block 保持主干形状不变",
      summary: "注意力与前馈网络都返回 [B,T,D]，残差连接才能逐元素相加。",
      paragraphs: [
        "一个 block 先规范化输入，经多头注意力和 dropout 后加回 shortcut；再规范化，经前馈网络和 dropout 后再次加回 shortcut。前馈层在每个 token 位置独立工作，通常先扩展特征宽度再投影回 D。",
        "多个 block 串联后，final norm 仍输出 [B,T,D]。无 bias 的输出线性层把最后一维 D 映射为词表 V，得到 [B,T,V] logits。一次 forward 不生成 token；生成循环从最后位置 [B,V] 选一个 id，再追加到输入。",
      ],
      bullets: [
        "残差路径保留原输入，并要求分支输出与 shortcut 形状相同。",
        "LayerNorm 改变数值分布，不改变 B、T、D。",
        "训练使用全部 T 个位置；逐 token 生成通常只读取最后位置 logits。",
      ],
      code: {
        title: "GPTModel 的完整形状主线",
        language: "python",
        snippet: `def forward(self, in_idx):
    batch_size, seq_len = in_idx.shape
    tok_embeds = self.tok_emb(in_idx)
    pos_embeds = self.pos_emb(
        torch.arange(seq_len, device=in_idx.device)
    )
    x = tok_embeds + pos_embeds
    x = self.drop_emb(x)
    x = self.trf_blocks(x)
    x = self.final_norm(x)
    logits = self.out_head(x)
    return logits`,
        sourceUrl: `${upstream}/ch04/01_main-chapter-code/gpt.py#L198-L209`,
        sourceLabel: "ch04/01_main-chapter-code/gpt.py",
        note: "官方源码原文；仅把 position embedding 调用换行以便显示。",
      },
      visual: {
        title: "残差数据流",
        html: `<div class="lfs-residual" role="img" aria-label="两个残差分支保持 B T D 形状">
          <span>x [B,T,D]</span><i>LayerNorm</i><i>Attention</i><b>＋ x</b><i>LayerNorm</i><i>FFN</i><b>＋ shortcut</b><span>out [B,T,D]</span>
        </div>`,
        caption: "主干宽度不变；只有最终 output head 把 D 改为 V。",
      },
    },
    {
      id: "training-to-chat",
      title: "5. 预训练学下一 token，指令微调改变数据分布",
      summary: "两阶段都可使用 next-token loss；差别主要来自文本格式、样本来源和被计入 loss 的位置。",
      paragraphs: [
        "预训练把连续文本切成输入与右移目标，模型在大量上下文上学习语言统计。仓库的 calc_loss_batch 将 [B,T,V] logits 展平为 [B×T,V]，把 [B,T] targets 展平为 [B×T]，再计算交叉熵。",
        "指令微调先把 instruction、可选 input 和 response 格式化为一段文本。自定义 collate 函数追加 end-of-text、按 batch 最长样本 padding，并把重复 padding target 改为 −100，使 PyTorch 交叉熵忽略这些位置。它仍是 token 级训练，不是单独的对话 API。",
        "评估需要把训练 loss 与回答质量分开看。仓库生成测试集回答后，可用本地模型或外部 LLM 对回答打分；这种评分依赖评审模型、提示和样本，应保留测试数据并检查具体输出。",
      ],
      bullets: [
        "预训练目标提供通用续写能力；指令数据让模型适配指令—回答格式。",
        "padding 位置不应参与 loss，否则模型会被训练去预测填充 token。",
        "低 validation loss 不等同于回答事实正确，需要任务级样本评估。",
      ],
      code: {
        title: "指令 batch 的右移目标与 padding mask",
        language: "python",
        snippet: `new_item += [pad_token_id]
padded = new_item + [pad_token_id] * (
    batch_max_length - len(new_item)
)
inputs = torch.tensor(padded[:-1])
targets = torch.tensor(padded[1:])

mask = targets == pad_token_id
indices = torch.nonzero(mask).squeeze()
if indices.numel() > 1:
    targets[indices[1:]] = ignore_index`,
        sourceUrl: `${upstream}/ch07/01_main-chapter-code/gpt_instruction_finetuning.py#L69-L83`,
        sourceLabel: "ch07/01_main-chapter-code/gpt_instruction_finetuning.py",
        note: "官方源码原文；注释省略，padding 表达式仅换行。ignore_index 默认值为 −100。",
      },
      visual: {
        title: "同一模型的两阶段数据路径",
        html: `<div class="lfs-stages" role="img" aria-label="预训练与指令微调数据路径">
          <div><b>预训练</b><span>连续文本</span><i>→</i><span>右移目标</span><i>→</i><strong>next-token loss</strong></div>
          <div><b>指令微调</b><span>instruction + input + response</span><i>→</i><span>padding / ignore</span><i>→</i><strong>next-token loss</strong></div>
          <div><b>评估</b><span>未见指令</span><i>→</i><span>生成回答</span><i>→</i><strong>逐样本检查 / 评分</strong></div>
        </div>`,
        caption: "模型结构可以不变；数据组织和评估协议发生变化。",
      },
    },
  ],
  lab: {
    title: "注意力与 shape 追踪器",
    intro: "编辑最多 6 个空格分隔 token，调整 embedding 维度和 head 数，逐步观察一个 batch 的 token ids、Q/K/V、第一头注意力矩阵、残差、logits 与 next-token loss。数值由浏览器确定性生成，只用于解释机制。",
    html: `<div class="lfs-lab" data-lfs-lab>
      <form class="lfs-controls" data-controls>
        <label class="lfs-token-input"><span>Tokens（空格分隔，最多 6 个）</span><input data-tokens type="text" value="机器 学习 从 数据 开始" autocomplete="off" spellcheck="false"></label>
        <label><span>Embedding D <output data-d-value>8</output></span><input data-dim type="range" min="4" max="12" step="4" value="8"></label>
        <label><span>Heads H</span><select data-heads><option value="1">1</option><option value="2" selected>2</option><option value="4">4</option></select></label>
        <label class="lfs-switch"><input data-causal type="checkbox" checked><span>Causal mask</span></label>
        <button type="submit">应用参数</button>
      </form>

      <div class="lfs-stepbar" aria-label="forward pass 步骤">
        <button type="button" data-prev aria-label="上一步">←</button>
        <div data-stages role="list"></div>
        <button type="button" data-next aria-label="下一步">→</button>
      </div>

      <div class="lfs-lab-summary">
        <div><span>当前步骤</span><strong data-step-name>Token IDs</strong></div>
        <div><span>序列长度 T</span><strong data-token-count>5</strong></div>
        <div><span>每头宽度 dₕ</span><strong data-head-dim>4</strong></div>
        <div><span>toy loss</span><strong data-loss>—</strong></div>
      </div>

      <section class="lfs-panel" aria-labelledby="lfs-shapes-title">
        <div class="lfs-panel-head"><h3 id="lfs-shapes-title">Shape ledger</h3><span>batch B = 1 · toy vocab V = 32</span></div>
        <div class="lfs-shape-ledger" data-shapes></div>
      </section>

      <div class="lfs-lab-grid">
        <section class="lfs-panel" aria-labelledby="lfs-matrix-title">
          <div class="lfs-panel-head"><h3 id="lfs-matrix-title">Head 0 attention</h3><span data-matrix-mode>mask 后概率</span></div>
          <div class="lfs-matrix-tabs" role="group" aria-label="矩阵显示模式">
            <button type="button" data-matrix="score">raw score</button>
            <button type="button" data-matrix="prob" class="is-active">probability</button>
          </div>
          <div class="lfs-heatmap" data-heatmap role="table" aria-label="第一注意力头矩阵"></div>
          <div class="lfs-legend"><span>低</span><i></i><span>高</span><b>× masked</b></div>
        </section>

        <section class="lfs-panel" aria-labelledby="lfs-flow-title">
          <div class="lfs-panel-head"><h3 id="lfs-flow-title">Residual data flow</h3><span>保持 [1,T,D]</span></div>
          <div class="lfs-flow" data-flow></div>
          <div class="lfs-vector" data-vector aria-label="最后一个 token 的隐藏向量"></div>
          <ol class="lfs-logits" data-logits aria-label="最后位置的 top logits"></ol>
        </section>
      </div>

      <div class="lfs-shortcuts"><kbd>←</kbd><kbd>→</kbd> 切换步骤 · <kbd>Home</kbd> 回到 token ids · <kbd>End</kbd> 到 loss</div>
      <p class="lfs-live" data-live aria-live="polite"></p>
    </div>`,
  },
  takeaways: [
    "始终先写形状：token ids [B,T] → hidden states [B,T,D] → logits [B,T,V]。",
    "多头注意力把 D 拆为 H×dₕ，score 和 probability 的形状是 [B,H,T,T]。",
    "causal mask 让第 t 个位置只读取 0…t，从而支持并行 next-token 训练。",
    "Transformer block 的注意力、前馈与残差分支都回到 [B,T,D]。",
    "指令微调继续使用 token 级目标，但改变文本格式、padding mask 和评估方式。",
  ],
  sources: [
    { label: "LLMs-from-scratch 官方仓库", url: "https://github.com/rasbt/LLMs-from-scratch" },
    { label: "第 4 章汇总代码：GPT 模型", url: `${upstream}/ch04/01_main-chapter-code/gpt.py` },
    { label: "第 5 章训练代码", url: `${upstream}/ch05/01_main-chapter-code/gpt_train.py` },
    { label: "第 7 章指令微调代码", url: `${upstream}/ch07/01_main-chapter-code/gpt_instruction_finetuning.py` },
    { label: "仓库 README 与章节索引", url: `${upstream}/README.md` },
  ],
};
