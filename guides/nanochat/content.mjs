const upstreamRef = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd";
const source = (path) => `https://github.com/karpathy/nanochat/blob/${upstreamRef}/${path}`;

export default {
  slug: "nanochat",
  title: "nanochat",
  subtitle: "从 tokenizer 训练到聊天推理的单节点大语言模型实验框架。",
  repoUrl: "https://github.com/karpathy/nanochat",
  accent: "#e85d2a",
  tags: ["End-to-end LLM", "Pretraining", "SFT", "Evaluation"],
  audience: "读过 nanoGPT，想把预训练扩展为 tokenizer、后训练、评估与推理完整实验的读者。",
  prerequisites: [
    "理解 Transformer、next-token prediction 与交叉熵",
    "知道 BPE tokenizer、预训练和监督微调的作用",
    "能阅读 PyTorch 训练循环与 shell 运行脚本",
  ],
  outcomes: [
    "沿 speedrun.sh 追踪数据、tokenizer、base checkpoint 与 SFT checkpoint",
    "解释 depth 如何派生宽度、注意力头数与训练 token horizon",
    "区分 base BPB、CORE 与 SFT 后 ChatCORE 的评估对象",
    "用教学规划器比较模型深度、GPU 数、时长与预算",
  ],
  overview: [
    "nanoGPT 主要展示 GPT 预训练；nanochat 把同一类可读代码扩展成一条端到端流水线。仓库内包含数据下载、BPE tokenizer、base pretraining、base evaluation、SFT、chat evaluation、推理引擎和 CLI。",
    "参考流程集中在 runs/speedrun.sh。它先准备 ClimbMix 数据并训练 tokenizer，再训练 base model、计算 BPB 与 CORE，随后从 base checkpoint 继续 SFT，最后评估并通过 CLI 对话。各阶段通过本地缓存目录中的 tokenizer 和 checkpoint 衔接。",
    "仓库把 depth 作为主要规模参数。base_train.py 由 depth 计算 model dimension 与 attention heads，并根据参数规模估算训练 token 数、batch size、学习率和 weight decay。交互实验只复现这些关系的简化教学模型，不执行训练，也不代表官方 benchmark。",
  ],
  architecture: {
    title: "一条流水线，两类 checkpoint",
    description: "点击节点查看产物如何流动：tokenizer 供所有后续阶段编码；base checkpoint 进入评估和 SFT；SFT checkpoint 再进入 chat eval 与推理。",
    nodes: [
      { id: "data", label: "ClimbMix shards", detail: "nanochat.dataset 下载 parquet；最后一个 shard 固定作 validation。" },
      { id: "tokenizer", label: "BPE tokenizer", detail: "tok_train.py 从训练文本学习 32,768 词表并保存 token_bytes。" },
      { id: "pretrain", label: "Base pretraining", detail: "base_train.py 在线 tokenize 文本，训练 next-token prediction 模型。" },
      { id: "base-ckpt", label: "Base checkpoint", detail: "保存模型、每个 rank 的 optimizer state、配置与 dataloader state。" },
      { id: "base-eval", label: "BPB · CORE · sample", detail: "base_eval.py 衡量压缩损失、任务能力并生成未对齐样本。" },
      { id: "sft", label: "Chat SFT", detail: "chat_sft.py 混合 SmolTalk、MMLU 与 GSM8K，并继承 base 配置。" },
      { id: "sft-ckpt", label: "SFT checkpoint", detail: "保存经过对话格式和任务数据微调的模型。" },
      { id: "chat", label: "Eval · CLI", detail: "chat_eval.py 评估对话任务；chat_cli.py 用 Engine 维护多轮 token 状态。" },
    ],
    edges: [
      { from: "data", to: "tokenizer", label: "text iterator" },
      { from: "data", to: "pretrain", label: "train / val" },
      { from: "tokenizer", to: "pretrain", label: "encode" },
      { from: "pretrain", to: "base-ckpt", label: "save" },
      { from: "base-ckpt", to: "base-eval", label: "load base" },
      { from: "base-ckpt", to: "sft", label: "warm start" },
      { from: "tokenizer", to: "sft", label: "render chat" },
      { from: "sft", to: "sft-ckpt", label: "save" },
      { from: "sft-ckpt", to: "chat", label: "load sft" },
    ],
  },
  sections: [
    {
      id: "pipeline",
      title: "从脚本读端到端流程",
      summary: "speedrun.sh 是阶段顺序和命令参数的可执行说明。",
      paragraphs: [
        "脚本先下载 8 个 shard 供 tokenizer 训练，同时在后台继续下载预训练所需数据。数据下载与 tokenizer 训练发生重叠，因此逐行相加命令耗时并不等于端到端墙钟时间。",
        "base_train 结束后先运行 base_eval，再执行 chat_sft 与 chat_eval。CLI 命令被保留为注释，因为它是训练后的交互入口，不应阻塞自动脚本。把每一段命令单独运行，可以替换深度、模型 tag 或评估规模。",
      ],
      bullets: [
        "Tokenizer：固定文本到 token id 的映射；更换 tokenizer 后，旧 checkpoint 不再直接兼容。",
        "Base：从网页文本学习 next-token prediction，产出未对齐的基础模型。",
        "SFT：从 base checkpoint 开始，用对话和任务数据训练 assistant 行为。",
        "Eval 与 chat：分别读取 base 或 SFT checkpoint，不产生新的训练阶段。",
      ],
      code: {
        title: "参考流水线中的阶段命令",
        language: "bash",
        snippet: `python -m nanochat.dataset -n 8
python -m scripts.tok_train
python -m scripts.tok_eval

torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \\
  --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- \\
  --device-batch-size=16
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft`,
        sourceUrl: source("runs/speedrun.sh"),
        sourceLabel: "runs/speedrun.sh",
        note: "按阶段摘录并省略日志参数；命令和路径来自当前官方脚本。",
      },
      visual: {
        title: "训练与使用不是一个循环",
        html: `<div class="nanochat-pipeline-strip" role="img" aria-label="tokenizer、pretrain、SFT、evaluation 与 chat 的顺序">
          <div><span>01</span><b>tokenizer</b><small>text → ids</small></div><i>→</i>
          <div><span>02</span><b>pretrain</b><small>base model</small></div><i>→</i>
          <div><span>03</span><b>SFT</b><small>assistant format</small></div><i>→</i>
          <div><span>04</span><b>evaluation</b><small>BPB · CORE</small></div><i>→</i>
          <div><span>05</span><b>chat</b><small>Engine · CLI</small></div>
        </div>`,
        caption: "评估会在 base 和 SFT 后分别发生；此处把它压成一条阅读路径。",
      },
    },
    {
      id: "tokenizer-data",
      title: "Tokenizer 先定义模型看到的单位",
      summary: "预训练 shard 提供文本，tok_train.py 用受限文本迭代器训练 BPE 并保存词表与 byte 计数。",
      paragraphs: [
        "nanochat.dataset 把 ClimbMix parquet 放到 base_data_climbmix。训练 split 使用除最后一个文件外的 shard，validation 使用最后一个 shard。tok_train.py 默认最多读取 20 亿字符，每篇文档最多取 10,000 字符，默认词表大小为 32,768。",
        "训练结果写入 tokenizer 目录。除 tokenizer 文件外，脚本还保存每个 token 对应的 byte 数；BPB 用它把 token-level loss 换算成对词表大小不敏感的 bits per byte。base_train 的 dataloader 在读取文档后再进行 tokenization 和 packing。",
      ],
      bullets: [
        "数据准备产物是 parquet 文本 shard，不是一次性离线 token 文件。",
        "tokenizer 同时服务于 pretraining、SFT 的 conversation rendering 和 inference decode。",
        "tokenizer eval 关注压缩率；base eval 的 BPB 关注模型在固定 tokenizer 下的预测损失。",
      ],
      code: {
        title: "限制字符数的 tokenizer 文本迭代器",
        language: "python",
        snippet: `def text_iterator():
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[:args.doc_cap]
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:
                return`,
        sourceUrl: source("scripts/tok_train.py"),
        sourceLabel: "scripts/tok_train.py",
        note: "官方实现原样摘录；默认 max_chars、doc_cap 与 vocab_size 在同一文件的 CLI 参数中定义。",
      },
      visual: {
        title: "同一 tokenizer 贯穿三种数据形态",
        html: `<div class="nanochat-token-flow" role="img" aria-label="网页文本和对话数据经过同一 tokenizer 进入模型">
          <div><span>raw text</span><b>ClimbMix parquet</b><small>训练 / 验证 shard</small></div>
          <strong>32,768 BPE<br><small>tokenizer + token_bytes</small></strong>
          <div><span>token rows</span><b>base dataloader</b><small>BOS best-fit packing</small></div>
          <div><span>conversation</span><b>SFT task mixture</b><small>role tokens + loss mask</small></div>
          <div><span>prompt</span><b>Engine</b><small>encode / generate / decode</small></div>
        </div>`,
        caption: "改变词表会改变 token id 与 checkpoint 的输入输出维度，因此要从 tokenizer 阶段重新开始。",
      },
    },
    {
      id: "depth-dial",
      title: "Depth 是规模旋钮，不只是层数",
      summary: "base_train.py 从 depth 派生宽度和头数，再由参数规模决定训练 horizon 与 batch 相关参数。",
      paragraphs: [
        "默认 aspect ratio 为 64，目标 head dimension 为 128。model dimension 先取 depth × 64，再向上取到 128 的倍数；attention head 数等于 model dimension ÷ 128。于是 d12 对应 768 维和 6 个头，d20 对应 1,280 维和 10 个头，d24 对应 1,536 维和 12 个头。",
        "模型建立后，base_train.py 统计 transformer matrices 加 lm_head 的 scaling parameters。默认训练 token horizon 是 scaling parameters 的 12 倍；当前 speedrun.sh 为 d24 显式使用 8 倍，以针对 GPT-2 能力阈值缩短训练。自动 batch、学习率与 weight decay 也会根据目标 token 数相对 d12 参考点调整。",
      ],
      bullets: [
        "depth 增加会同时增加 block 数与 hidden width，所以参数量不是随 depth 线性增长。",
        "更大模型通常需要降低 device batch size；程序用 gradient accumulation 补足 total batch tokens。",
        "同一个 depth 在不同 GPU、dtype、attention kernel 和数据上不会得到固定墙钟时间。",
      ],
      code: {
        title: "由 depth 构造模型并计算 token horizon",
        language: "python",
        snippet: `base_dim = depth * args.aspect_ratio
model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
num_heads = model_dim // args.head_dim
config = GPTConfig(
    sequence_len=args.max_seq_len, vocab_size=vocab_size,
    n_layer=depth, n_head=num_heads, n_kv_head=num_heads,
    n_embd=model_dim, window_pattern=args.window_pattern,
)

num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params)`,
        sourceUrl: source("scripts/base_train.py"),
        sourceLabel: "scripts/base_train.py",
        note: "合并同一文件中 build_model_meta 与 scaling-law 配置的相邻逻辑；未改变公式。",
      },
      visual: {
        title: "depth 同时拉长和加宽网络",
        html: `<div class="nanochat-depth-ruler" role="img" aria-label="depth 12、20 和 24 对应的宽度与注意力头数">
          <div style="--nanochat-scale:.5"><span>d12</span><b>768 dim</b><small>6 heads</small><i></i></div>
          <div style="--nanochat-scale:.78"><span>d20</span><b>1,280 dim</b><small>10 heads</small><i></i></div>
          <div style="--nanochat-scale:1"><span>d24</span><b>1,536 dim</b><small>12 heads</small><i></i></div>
        </div>`,
        caption: "条长表示层数，条高表示宽度；仅展示 base_train.py 默认 aspect ratio 与 head dimension。",
      },
    },
    {
      id: "pretrain-eval",
      title: "Time-to-GPT-2 是受控预训练实验",
      summary: "它比较固定硬件上超过同一 CORE 阈值所需的训练墙钟时间，不等于完整建站式流水线耗时。",
      paragraphs: [
        "base_train 在 next-token prediction 上优化，并记录 val_bpb、CORE、tokens/sec、MFU 与 total_training_time。base_eval 可独立运行 core、bpb、sample 三类评估：BPB 描述每 byte 的压缩损失，CORE 汇总 DCLM 的 22 个任务，sample 用相同 base checkpoint 生成文本。",
        "官方 leaderboard 把 8×H100 作为硬件条件，以 CORE 超过 GPT-2 1.6B 的 0.256525 为门槛，报告训练 iteration 的 total_training_time；评估与日志时间不计入。当前参考脚本使用 d24、FP8 和 8 倍 token-to-scaling-parameter ratio，但 leaderboard 结果会随提交变化。",
        "实验改动需要同时看 CORE、val_bpb 和吞吐。CORE 有运行噪声；数据集变化后 val_bpb 也不能直接与旧数据分布比较。速度提高但未过阈值，不构成更短的 time-to-GPT-2。",
      ],
      bullets: [
        "控制变量：代码提交、数据顺序、depth、dtype、batch 和 8×H100 节点。",
        "目标变量：达到 CORE > 0.256525 时的训练 wall-clock。",
        "辅助变量：val_bpb、tokens/sec、MFU、训练 FLOPs 与多次运行的分布。",
      ],
      code: null,
      visual: {
        title: "先过能力线，再比较时间",
        html: `<div class="nanochat-race-chart" role="img" aria-label="不同实验只有超过 CORE 阈值后才能比较训练时间">
          <div class="nanochat-race-threshold"><span>CORE 0.256525</span></div>
          <div><b>run A</b><i style="--nanochat-core:72%"></i><span>未过线</span></div>
          <div><b>run B</b><i style="--nanochat-core:91%"></i><span>过线 · 105 min</span></div>
          <div><b>run C</b><i style="--nanochat-core:94%"></i><span>过线 · 99 min</span></div>
        </div>`,
        caption: "长度是阈值附近的示意，不对应真实 CORE 比例；时间只有在过线后才进入 leaderboard 比较。",
      },
    },
    {
      id: "sft-chat",
      title: "SFT 改变训练数据与损失位置",
      summary: "chat_sft 从 base checkpoint 继承模型和多数训练配置，再用对话 mixture 训练 assistant completion。",
      paragraphs: [
        "chat_sft.py 载入 base 模型，并在未显式覆盖时继承 context length、device batch、total batch 和学习率。训练数据混合 SmolTalk、MMLU 与 GSM8K；tokenizer.render_conversation 插入角色特殊 token，并返回 loss mask，使损失集中在需要模型生成的部分。",
        "SFT 产物写到 chatsft_checkpoints。chat_eval 以 sft 为输入评估对话任务；chat_cli 同样加载 sft checkpoint，使用 Engine 生成，并把 user/assistant 边界 token 追加到 conversation state。清空对话只重置 token history，不修改模型。",
      ],
      bullets: [
        "base checkpoint 学到语言与知识；SFT checkpoint 在其上增加对话格式和任务行为。",
        "模型结构与 tokenizer 必须兼容，SFT 不是从随机参数重新训练。",
        "CLI 的 temperature 与 top-k 改变采样，不改变 checkpoint。",
      ],
      code: {
        title: "从 base 加载并构造 SFT mixture",
        language: "python",
        snippet: `model, tokenizer, meta = load_model(
    "base", device, phase="train",
    model_tag=args.model_tag, step=args.model_step,
)

train_tasks = [
    SmolTalk(split="train"),
    *[MMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)],
    *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)],
]
train_dataset = TaskMixture(train_tasks)`,
        sourceUrl: source("scripts/chat_sft.py"),
        sourceLabel: "scripts/chat_sft.py",
        note: "从模型加载和 SFT data mixture 两处摘录并合并；任务名称与 split 保持原样。",
      },
      visual: {
        title: "Checkpoint 是阶段边界",
        html: `<div class="nanochat-checkpoint-map" role="img" aria-label="base checkpoint 分支到 base eval 和 SFT，SFT checkpoint 分支到 chat eval 和 CLI">
          <div class="is-train"><span>base_train</span><b>next-token loss</b></div>
          <i>save →</i><div class="is-checkpoint"><span>base checkpoint</span><b>model · optim · meta</b></div>
          <div class="is-branch"><span>base_eval</span><b>BPB · CORE · sample</b></div>
          <i>load →</i><div class="is-train"><span>chat_sft</span><b>conversation mask</b></div>
          <i>save →</i><div class="is-checkpoint"><span>SFT checkpoint</span><b>assistant model</b></div>
          <div class="is-branch"><span>chat_eval / CLI</span><b>measure · generate</b></div>
        </div>`,
        caption: "评估只读取 checkpoint；SFT 同时读取 base 参数、tokenizer 和训练元数据。",
      },
    },
  ],
  lab: {
    title: "端到端训练预算规划器",
    intro: "调整 depth、GPU 数、时长、预算与单价。规划器用当前代码中的宽度/头数公式和一个明确标注的简化参数模型，展示规模、训练覆盖度、阶段时间和 checkpoint 流向。",
    html: `<div class="nanochat-lab" data-nanochat-lab>
      <p class="nanochat-disclaimer"><b>教学估算</b> 参数量忽略少量 gate/scalar；时间以 d24、8 张同等级 GPU、99 分钟预训练为归一化参考，并假设理想线性多卡缩放。它不是 nanochat 官方 benchmark，也不能预测你的机器结果。</p>

      <div class="nanochat-controls" aria-label="训练预算参数">
        <label><span>depth <output data-nanochat-value="depth">24</output></span><input type="range" min="4" max="32" step="2" value="24" data-nanochat-input="depth"></label>
        <label><span>GPU 数 <output data-nanochat-value="gpus">8</output></span><input type="range" min="1" max="8" step="1" value="8" data-nanochat-input="gpus"></label>
        <label><span>计划时长 <output data-nanochat-value="hours">2.0 h</output></span><input type="range" min="0.5" max="48" step="0.5" value="2" data-nanochat-input="hours"></label>
        <label><span>预算 <output data-nanochat-value="budget">$48</output></span><input type="range" min="5" max="1000" step="5" value="48" data-nanochat-input="budget"></label>
        <label class="nanochat-price"><span>单卡单价 / 小时</span><span class="nanochat-money"><b>$</b><input type="number" min="0.1" max="30" step="0.1" value="3" inputmode="decimal" data-nanochat-input="price"></span></label>
      </div>

      <p class="nanochat-sync-note" data-nanochat-sync-note>当前按计划时长计算预算；拖动预算后会反算可用时长。</p>
      <p class="nanochat-live" aria-live="polite" aria-atomic="true" data-nanochat-live></p>

      <section class="nanochat-lab-panel" aria-labelledby="nanochat-model-heading">
        <header><span>01</span><div><h3 id="nanochat-model-heading">模型与训练 horizon</h3><p>width/head 使用当前 base_train.py 公式</p></div></header>
        <div class="nanochat-metrics" data-nanochat-metrics></div>
        <div class="nanochat-coverage" data-nanochat-coverage></div>
      </section>

      <section class="nanochat-lab-panel" aria-labelledby="nanochat-stage-heading">
        <header><span>02</span><div><h3 id="nanochat-stage-heading">阶段时间分配</h3><p>总时长按教学比例拆分；真实下载可与 tokenizer 重叠</p></div></header>
        <div class="nanochat-stage-bar" data-nanochat-stage-bar role="img" aria-label="阶段时间比例"></div>
        <div class="nanochat-stage-list" data-nanochat-stage-list></div>
      </section>

      <section class="nanochat-lab-panel" aria-labelledby="nanochat-flow-heading">
        <header><span>03</span><div><h3 id="nanochat-flow-heading">逐步检查流水线</h3><p>点击阶段或使用左右方向键</p></div></header>
        <div class="nanochat-stepper" role="group" aria-label="nanochat 流水线阶段" data-nanochat-stepper>
          <button type="button" data-nanochat-step="0"><span>01</span><b>tokenizer</b><small>text → ids</small></button>
          <button type="button" data-nanochat-step="1"><span>02</span><b>pretrain</b><small>base checkpoint</small></button>
          <button type="button" data-nanochat-step="2"><span>03</span><b>SFT</b><small>chat checkpoint</small></button>
          <button type="button" data-nanochat-step="3"><span>04</span><b>eval</b><small>BPB · CORE</small></button>
          <button type="button" data-nanochat-step="4"><span>05</span><b>chat</b><small>Engine · CLI</small></button>
        </div>
        <div class="nanochat-flow-detail" data-nanochat-flow-detail></div>
      </section>
    </div>`,
  },
  takeaways: [
    "把 runs/speedrun.sh 当作端到端索引，再进入每个 scripts 文件查看阶段内部。",
    "tokenizer 是跨阶段依赖；更换词表后，base 与 SFT checkpoint 都需要重新生成。",
    "depth 通过宽度、头数、参数量和 token horizon 同时改变模型与训练预算。",
    "base evaluation 测 BPB、CORE 和 sample；SFT 后的 chat evaluation 测不同对象。",
    "time-to-GPT-2 固定 8×H100 与 CORE 阈值，只统计训练 iteration 时间。",
    "页面规划器用于理解变量关系，不替代在目标硬件上测 tokens/sec、显存和评估分数。",
  ],
  sources: [
    { label: "nanochat README", url: source("README.md") },
    { label: "参考端到端流程", url: source("runs/speedrun.sh") },
    { label: "Tokenizer 训练", url: source("scripts/tok_train.py") },
    { label: "预训练与 depth scaling", url: source("scripts/base_train.py") },
    { label: "Base evaluation", url: source("scripts/base_eval.py") },
    { label: "SFT 训练", url: source("scripts/chat_sft.py") },
    { label: "Chat evaluation", url: source("scripts/chat_eval.py") },
    { label: "CLI 与推理入口", url: source("scripts/chat_cli.py") },
    { label: "Checkpoint 保存与加载", url: source("nanochat/checkpoint_manager.py") },
    { label: "Time-to-GPT-2 规则", url: source("dev/LEADERBOARD.md") },
  ],
};
