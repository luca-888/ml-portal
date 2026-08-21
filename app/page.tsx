"use client";

import {
  ArrowUpRight,
  Bookmark,
  BookOpen,
  Boxes,
  Check,
  Compass,
  FolderGit2,
  GraduationCap,
  Grid2X2,
  Plus,
  Search,
  Sparkles,
  Wrench,
  X,
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

type Category = "repo" | "research" | "blog" | "course" | "tool";

type Resource = {
  id: string;
  category: Category;
  title: string;
  description: string;
  href: string;
  source: string;
  tags: string[];
  featured?: boolean;
  custom?: boolean;
};

const resources: Resource[] = [
  {
    id: "pytorch",
    category: "repo",
    title: "PyTorch",
    description: "灵活、成熟的深度学习框架；从研究原型到生产部署都值得作为默认入口。",
    href: "https://github.com/pytorch/pytorch",
    source: "pytorch/pytorch",
    tags: ["Framework", "Training"],
    featured: true,
  },
  {
    id: "hugging-face",
    category: "tool",
    title: "Hugging Face",
    description: "模型、数据集与 Demo 的开放协作中心，也是现代 ML 工作流的常驻工具箱。",
    href: "https://huggingface.co/",
    source: "huggingface.co",
    tags: ["Models", "Datasets"],
    featured: true,
  },
  {
    id: "vllm",
    category: "repo",
    title: "vLLM",
    description: "高吞吐、内存高效的大模型推理与服务引擎，做 serving 时绕不开。",
    href: "https://github.com/vllm-project/vllm",
    source: "vllm-project/vllm",
    tags: ["LLM", "Inference"],
    featured: true,
  },
  {
    id: "lilian-weng",
    category: "blog",
    title: "Lil’Log",
    description: "带推导、参考文献和清晰图解的长篇学习笔记，适合系统理解一个方向。",
    href: "https://lilianweng.github.io/",
    source: "lilianweng.github.io",
    tags: ["Deep Dive", "LLM"],
    featured: true,
  },
  {
    id: "transformers",
    category: "repo",
    title: "Transformers",
    description: "预训练模型生态的事实标准接口，覆盖文本、视觉、音频与多模态。",
    href: "https://github.com/huggingface/transformers",
    source: "huggingface/transformers",
    tags: ["LLM", "Multimodal"],
  },
  {
    id: "llama-cpp",
    category: "repo",
    title: "llama.cpp",
    description: "用 C/C++ 在广泛硬件上高效运行大模型，本地推理与量化实践的宝库。",
    href: "https://github.com/ggml-org/llama.cpp",
    source: "ggml-org/llama.cpp",
    tags: ["Local AI", "Inference"],
  },
  {
    id: "scikit-learn",
    category: "repo",
    title: "scikit-learn",
    description: "经典机器学习最可靠的工具箱之一，API 设计和文档本身也值得学习。",
    href: "https://github.com/scikit-learn/scikit-learn",
    source: "scikit-learn/scikit-learn",
    tags: ["Classical ML", "Python"],
  },
  {
    id: "liger-kernel",
    category: "repo",
    title: "Liger Kernel",
    description: "面向大模型训练的高效 Triton kernels，用更少显存争取更高吞吐。",
    href: "https://github.com/linkedin/Liger-Kernel",
    source: "linkedin/Liger-Kernel",
    tags: ["Systems", "Triton"],
  },
  {
    id: "mlx",
    category: "repo",
    title: "MLX",
    description: "Apple silicon 上的数组与机器学习框架，统一内存架构下的轻量实验利器。",
    href: "https://github.com/ml-explore/mlx",
    source: "ml-explore/mlx",
    tags: ["Apple Silicon", "Framework"],
  },
  {
    id: "trl",
    category: "repo",
    title: "TRL",
    description: "用强化学习、偏好优化等方法对 Transformer 模型做后训练的完整工具集。",
    href: "https://github.com/huggingface/trl",
    source: "huggingface/trl",
    tags: ["Post-training", "RLHF"],
  },
  {
    id: "arxiv-cslg",
    category: "research",
    title: "arXiv · cs.LG",
    description: "机器学习论文的每日源头。适合建立固定浏览节奏，而不是被二手摘要牵着走。",
    href: "https://arxiv.org/list/cs.LG/recent",
    source: "arxiv.org",
    tags: ["Papers", "Daily"],
  },
  {
    id: "papers-with-code",
    category: "research",
    title: "Papers with Code",
    description: "把论文、代码和 benchmark 连在一起，适合追踪一个任务的演进与实现。",
    href: "https://paperswithcode.com/",
    source: "paperswithcode.com",
    tags: ["Benchmarks", "Code"],
  },
  {
    id: "openreview",
    category: "research",
    title: "OpenReview",
    description: "阅读公开评审、作者回复与讨论，在结论之外理解研究争议和判断过程。",
    href: "https://openreview.net/",
    source: "openreview.net",
    tags: ["Peer Review", "Papers"],
  },
  {
    id: "semantic-scholar",
    category: "research",
    title: "Semantic Scholar",
    description: "适合从一篇关键论文出发，沿引用关系寻找前置工作与后续影响。",
    href: "https://www.semanticscholar.org/",
    source: "semanticscholar.org",
    tags: ["Discovery", "Citations"],
  },
  {
    id: "distill",
    category: "blog",
    title: "Distill",
    description: "机器学习可视化解释的经典档案；虽已停止更新，旧文依然极有价值。",
    href: "https://distill.pub/",
    source: "distill.pub",
    tags: ["Visual Essays", "Archive"],
  },
  {
    id: "jay-alammar",
    category: "blog",
    title: "Jay Alammar",
    description: "用高质量图解拆开 Transformer、BERT 等关键概念，入门和复习都好用。",
    href: "https://jalammar.github.io/",
    source: "jalammar.github.io",
    tags: ["Visual Guide", "NLP"],
  },
  {
    id: "sebastian-raschka",
    category: "blog",
    title: "Ahead of AI",
    description: "Sebastian Raschka 的研究与工程笔记，兼顾模型原理、训练细节和新论文。",
    href: "https://magazine.sebastianraschka.com/",
    source: "sebastianraschka.com",
    tags: ["LLM", "Engineering"],
  },
  {
    id: "chip-huyen",
    category: "blog",
    title: "Chip Huyen",
    description: "机器学习系统、数据与 AI 工程的实践思考，尤其适合关注真实生产约束的人。",
    href: "https://huyenchip.com/blog/",
    source: "huyenchip.com",
    tags: ["MLOps", "Systems"],
  },
  {
    id: "fastai",
    category: "course",
    title: "Practical Deep Learning",
    description: "fast.ai 的代码优先深度学习课程，从做出可用模型开始，再向下理解原理。",
    href: "https://course.fast.ai/",
    source: "fast.ai",
    tags: ["Hands-on", "Free"],
  },
  {
    id: "cs231n",
    category: "course",
    title: "Stanford CS231n",
    description: "计算机视觉与深度学习的经典课程，讲义、作业和推导都有长期参考价值。",
    href: "https://cs231n.stanford.edu/",
    source: "stanford.edu",
    tags: ["Vision", "Theory"],
  },
  {
    id: "full-stack-dl",
    category: "course",
    title: "Full Stack Deep Learning",
    description: "从问题定义、数据、训练一路走到部署与监控，补齐模型之外的完整链路。",
    href: "https://fullstackdeeplearning.com/",
    source: "fullstackdeeplearning.com",
    tags: ["Production", "MLOps"],
  },
  {
    id: "hf-learn",
    category: "course",
    title: "Hugging Face Learn",
    description: "围绕 LLM、agents、音频、视觉等方向组织的免费实践课程合集。",
    href: "https://huggingface.co/learn",
    source: "huggingface.co/learn",
    tags: ["LLM", "Hands-on"],
  },
  {
    id: "deep-learning-book",
    category: "course",
    title: "Deep Learning Book",
    description: "Goodfellow、Bengio 与 Courville 的经典教材，查概念和补理论的可靠底座。",
    href: "https://www.deeplearningbook.org/",
    source: "deeplearningbook.org",
    tags: ["Textbook", "Theory"],
  },
  {
    id: "wandb",
    category: "tool",
    title: "Weights & Biases",
    description: "实验跟踪、模型评估与协作平台，让训练过程从散落日志变成可比较记录。",
    href: "https://wandb.ai/site",
    source: "wandb.ai",
    tags: ["Experiments", "MLOps"],
  },
  {
    id: "gradio",
    category: "tool",
    title: "Gradio",
    description: "用很少的 Python 代码把模型变成交互界面，分享原型和内部评估尤其顺手。",
    href: "https://www.gradio.app/",
    source: "gradio.app",
    tags: ["Demo", "Python"],
  },
  {
    id: "kaggle",
    category: "tool",
    title: "Kaggle Notebooks",
    description: "免配置的 Notebook、公开数据集和社区案例，快速验证想法的实用起点。",
    href: "https://www.kaggle.com/code",
    source: "kaggle.com/code",
    tags: ["Notebooks", "Datasets"],
  },
];

const categoryMeta: Array<{
  key: "all" | Category;
  label: string;
  icon: typeof Grid2X2;
}> = [
  { key: "all", label: "全部资源", icon: Grid2X2 },
  { key: "repo", label: "开源仓库", icon: FolderGit2 },
  { key: "research", label: "论文研究", icon: BookOpen },
  { key: "blog", label: "博客文章", icon: Boxes },
  { key: "course", label: "课程学习", icon: GraduationCap },
  { key: "tool", label: "工具平台", icon: Wrench },
];

const categoryLabels: Record<Category, string> = {
  repo: "REPO",
  research: "RESEARCH",
  blog: "BLOG",
  course: "COURSE",
  tool: "TOOL",
};

const cardTones = ["cobalt", "amber", "coral", "green"];

function safeParseCustomResources(value: string | null): Resource[] {
  if (!value) return [];
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState<"all" | Category>("all");
  const [savedOnly, setSavedOnly] = useState(false);
  const [saved, setSaved] = useState<string[]>([]);
  const [customResources, setCustomResources] = useState<Resource[]>([]);
  const [modalOpen, setModalOpen] = useState(false);
  const [formError, setFormError] = useState("");
  const searchRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const frame = requestAnimationFrame(() => {
      setSaved(JSON.parse(localStorage.getItem("gradient-atlas:saved") || "[]"));
      setCustomResources(
        safeParseCustomResources(localStorage.getItem("gradient-atlas:custom")),
      );
    });
    return () => cancelAnimationFrame(frame);
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        searchRef.current?.focus();
      }
      if (event.key === "Escape") setModalOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const allResources = useMemo(
    () => [...customResources, ...resources],
    [customResources],
  );

  const filteredResources = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return allResources.filter((resource) => {
      const matchesCategory = category === "all" || resource.category === category;
      const matchesSaved = !savedOnly || saved.includes(resource.id);
      const searchable = [
        resource.title,
        resource.description,
        resource.source,
        ...resource.tags,
      ]
        .join(" ")
        .toLowerCase();
      return matchesCategory && matchesSaved && searchable.includes(normalized);
    });
  }, [allResources, category, query, saved, savedOnly]);

  const featured = resources.filter((resource) => resource.featured);
  const isDefaultView = category === "all" && !query && !savedOnly;

  const toggleSaved = (id: string) => {
    setSaved((current) => {
      const next = current.includes(id)
        ? current.filter((item) => item !== id)
        : [...current, id];
      localStorage.setItem("gradient-atlas:saved", JSON.stringify(next));
      return next;
    });
  };

  const openRandom = () => {
    if (!filteredResources.length) return;
    const resource =
      filteredResources[Math.floor(Math.random() * filteredResources.length)];
    window.open(resource.href, "_blank", "noopener,noreferrer");
  };

  const addResource = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setFormError("");
    const data = new FormData(event.currentTarget);
    const title = String(data.get("title") || "").trim();
    const href = String(data.get("href") || "").trim();
    const description = String(data.get("description") || "").trim();
    const nextCategory = String(data.get("category") || "repo") as Category;

    try {
      const parsed = new URL(href);
      if (!["http:", "https:"].includes(parsed.protocol)) throw new Error();
      if (!title) throw new Error();

      const next: Resource = {
        id: `custom-${Date.now()}`,
        title,
        href: parsed.toString(),
        description: description || "我想稍后认真看看的收藏。",
        category: nextCategory,
        source: parsed.hostname.replace(/^www\./, ""),
        tags: ["Personal"],
        custom: true,
      };
      const updated = [next, ...customResources];
      setCustomResources(updated);
      localStorage.setItem("gradient-atlas:custom", JSON.stringify(updated));
      event.currentTarget.reset();
      setModalOpen(false);
    } catch {
      setFormError("请填写名称，并输入一个有效的 http(s) 链接。 ");
    }
  };

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="梯度图谱首页">
          <span className="brand-mark">G/</span>
          <span>GRADIENT ATLAS</span>
        </a>
        <nav aria-label="主导航">
          <a className="active" href="#library">资料库</a>
          <a href="#about">关于</a>
          <a href="https://github.com/luca-888/ml-portal" target="_blank" rel="noreferrer">
            GitHub <ArrowUpRight size={13} aria-hidden="true" />
          </a>
        </nav>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <p className="eyebrow">A PERSONAL MACHINE LEARNING INDEX</p>
          <h1>
            把好奇心，整理成一张<span>可抵达的地图。</span>
          </h1>
          <p className="lede">精选 repo、论文、课程、博客与工具。只留下值得反复打开的东西。</p>
          <div className="hero-actions">
            <a className="primary-action" href="#library">
              开始探索 <ArrowUpRight size={17} aria-hidden="true" />
            </a>
            <button type="button" onClick={openRandom}>
              <Compass size={17} aria-hidden="true" /> 随便逛逛
            </button>
          </div>
        </div>
        <div className="hero-index" aria-label="资源统计">
          <div><strong>{String(allResources.length).padStart(2, "0")}</strong><span>精选收藏</span></div>
          <div><strong>05</strong><span>主题分类</span></div>
          <div><strong>∞</strong><span>持续生长</span></div>
        </div>
      </section>

      <section className="workspace" id="library">
        <aside className="rail">
          <p className="rail-label">INDEX / 01</p>
          <h2>浏览收藏</h2>
          <div className="filters" aria-label="资源类别">
            {categoryMeta.map((item, index) => {
              const Icon = item.icon;
              const count = item.key === "all"
                ? allResources.length
                : allResources.filter((resource) => resource.category === item.key).length;
              return (
                <button
                  className={category === item.key && !savedOnly ? "selected" : ""}
                  key={item.key}
                  type="button"
                  onClick={() => {
                    setCategory(item.key);
                    setSavedOnly(false);
                  }}
                >
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  <Icon size={15} aria-hidden="true" />
                  <b>{item.label}</b>
                  <em>{count}</em>
                </button>
              );
            })}
            <button
              className={savedOnly ? "selected" : ""}
              type="button"
              onClick={() => setSavedOnly((current) => !current)}
            >
              <span>07</span>
              <Bookmark size={15} aria-hidden="true" />
              <b>稍后阅读</b>
              <em>{saved.length}</em>
            </button>
          </div>
          <div className="quote-block">
            <span>NOTE /</span>
            <p>收藏不是终点。理解、实践、再回到这里。</p>
          </div>
        </aside>

        <div className="library">
          <div className="library-head">
            <label className="search-field">
              <Search size={20} aria-hidden="true" />
              <input
                ref={searchRef}
                type="search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="搜索 repo、主题或作者…"
                aria-label="搜索资源"
              />
              {query ? (
                <button type="button" onClick={() => setQuery("")} aria-label="清空搜索">
                  <X size={16} />
                </button>
              ) : (
                <kbd>⌘ K</kbd>
              )}
            </label>
            <button className="add-button" type="button" onClick={() => setModalOpen(true)}>
              <Plus size={16} aria-hidden="true" /> 添加收藏
            </button>
          </div>

          {isDefaultView && (
            <>
              <div className="section-title">
                <div>
                  <p>EDITOR’S PICK</p>
                  <h2>今天值得打开的四个入口</h2>
                </div>
                <span>01 — 04</span>
              </div>

              <div className="resource-grid featured-grid">
                {featured.map((resource, index) => (
                  <article className={`resource-card ${cardTones[index]}`} key={resource.id}>
                    <div className="card-top">
                      <span>{categoryLabels[resource.category]}</span>
                      <button
                        className={saved.includes(resource.id) ? "bookmarked" : ""}
                        type="button"
                        onClick={() => toggleSaved(resource.id)}
                        aria-label={saved.includes(resource.id) ? `取消收藏 ${resource.title}` : `收藏 ${resource.title}`}
                      >
                        {saved.includes(resource.id) ? <Check size={16} /> : <Bookmark size={16} />}
                      </button>
                    </div>
                    <div className="card-body">
                      <h3><a href={resource.href} target="_blank" rel="noreferrer">{resource.title}</a></h3>
                      <p>{resource.description}</p>
                    </div>
                    <a className="card-foot" href={resource.href} target="_blank" rel="noreferrer">
                      <span>{resource.source}</span>
                      <ArrowUpRight size={20} aria-hidden="true" />
                    </a>
                  </article>
                ))}
              </div>
            </>
          )}

          <div className="section-title archive-title">
            <div>
              <p>{isDefaultView ? "THE ARCHIVE" : "FILTERED VIEW"}</p>
              <h2>{isDefaultView ? "完整索引" : `${filteredResources.length} 个匹配结果`}</h2>
            </div>
            <span>{String(filteredResources.length).padStart(2, "0")} ITEMS</span>
          </div>

          {filteredResources.length ? (
            <div className="archive-list">
              {filteredResources.map((resource, index) => (
                <article className="archive-item" key={resource.id}>
                  <span className="item-number">{String(index + 1).padStart(2, "0")}</span>
                  <div className="item-main">
                    <div className="item-heading">
                      <span>{categoryLabels[resource.category]}</span>
                      {resource.custom && <span className="personal-label">PERSONAL</span>}
                    </div>
                    <h3><a href={resource.href} target="_blank" rel="noreferrer">{resource.title}</a></h3>
                    <p>{resource.description}</p>
                    <div className="tags">
                      {resource.tags.map((tag) => <span key={tag}>{tag}</span>)}
                    </div>
                  </div>
                  <div className="item-actions">
                    <button
                      className={saved.includes(resource.id) ? "bookmarked" : ""}
                      type="button"
                      onClick={() => toggleSaved(resource.id)}
                      aria-label={saved.includes(resource.id) ? `取消收藏 ${resource.title}` : `收藏 ${resource.title}`}
                      title="稍后阅读"
                    >
                      {saved.includes(resource.id) ? <Check size={17} /> : <Bookmark size={17} />}
                    </button>
                    <a href={resource.href} target="_blank" rel="noreferrer" aria-label={`打开 ${resource.title}`}>
                      <ArrowUpRight size={19} />
                    </a>
                  </div>
                </article>
              ))}
            </div>
          ) : (
            <div className="empty-state">
              <Sparkles size={24} aria-hidden="true" />
              <h3>这里暂时还是空白</h3>
              <p>换一个关键词，或者把你找到的好东西加进来。</p>
              <button type="button" onClick={() => { setQuery(""); setCategory("all"); setSavedOnly(false); }}>
                清除筛选
              </button>
            </div>
          )}
        </div>
      </section>

      <section className="about" id="about">
        <p>ABOUT / 02</p>
        <blockquote>“真正有用的收藏，应该让下一次寻找变得更短。”</blockquote>
        <div>
          <p>Gradient Atlas 是一个持续生长的个人 ML 索引。公开资源是起点，你在本机添加与标记的内容只保存在当前浏览器里。</p>
          <a href="https://github.com/luca-888/ml-portal" target="_blank" rel="noreferrer">
            在 GitHub 上一起维护 <ArrowUpRight size={15} />
          </a>
        </div>
      </section>

      <footer>
        <span>CURATED WITH INTENT</span>
        <span>GRADIENT ATLAS · 2026</span>
      </footer>

      {modalOpen && (
        <div className="modal-layer">
          <button className="modal-dismiss" type="button" aria-label="关闭添加收藏弹窗" onClick={() => setModalOpen(false)} />
          <section className="modal" role="dialog" aria-modal="true" aria-labelledby="add-title">
            <div className="modal-head">
              <div>
                <p>NEW ENTRY</p>
                <h2 id="add-title">添加一条私人收藏</h2>
              </div>
              <button type="button" onClick={() => setModalOpen(false)} aria-label="关闭弹窗"><X size={20} /></button>
            </div>
            <form onSubmit={addResource}>
              <label>名称<input name="title" required placeholder="例如：A Visual Guide to Mamba" /></label>
              <label>链接<input name="href" type="url" required placeholder="https://…" /></label>
              <label>分类
                <select name="category" defaultValue="repo">
                  {categoryMeta.slice(1).map((item) => <option value={item.key} key={item.key}>{item.label}</option>)}
                </select>
              </label>
              <label>一句话备注<textarea name="description" rows={3} placeholder="为什么值得留下？" /></label>
              {formError && <p className="form-error" role="alert">{formError}</p>}
              <div className="form-note">这条收藏仅保存在你的浏览器中。</div>
              <button className="submit-button" type="submit">保存到图谱 <ArrowUpRight size={16} /></button>
            </form>
          </section>
        </div>
      )}
    </main>
  );
}
