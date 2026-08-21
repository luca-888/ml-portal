import resources from "@/data/resources.json";

const categories = [
  { key: "repo", label: "开源仓库", number: "01" },
  { key: "course", label: "课程", number: "02" },
  { key: "research", label: "论文与研究", number: "03" },
  { key: "blog", label: "博客", number: "04" },
  { key: "tool", label: "工具", number: "05" },
] as const;

export default function Home() {
  return (
    <main>
      <header className="site-header">
        <a className="brand" href="#top">Gradient Atlas</a>
        <nav aria-label="页面导航">
          <a href="#resources">资源</a>
          <a href="https://github.com/luca-888/ml-portal" target="_blank" rel="noreferrer">
            GitHub ↗
          </a>
        </nav>
      </header>

      <section className="hero" id="top">
        <p className="eyebrow">A CURATED ML LEARNING LIST</p>
        <h1>机器学习，<br />从这些资源开始。</h1>
        <div className="hero-meta">
          <p>
            一份简单、持续维护的 ML 学习资源清单。仓库是唯一数据源，
            网站会根据清单自动生成。
          </p>
          <span>{resources.length} RESOURCES · {categories.length} CATEGORIES</span>
        </div>
      </section>

      <section className="catalog" id="resources">
        <aside>
          <p>按类别浏览</p>
          <nav aria-label="资源分类">
            {categories.map((category) => (
              <a href={`#${category.key}`} key={category.key}>
                <span>{category.number}</span>
                {category.label}
              </a>
            ))}
          </nav>
        </aside>

        <div className="resource-sections">
          {categories.map((category) => {
            const items = resources.filter((resource) => resource.category === category.key);

            return (
              <section className="resource-section" id={category.key} key={category.key}>
                <div className="section-heading">
                  <span>{category.number}</span>
                  <h2>{category.label}</h2>
                  <em>{items.length}</em>
                </div>

                <div className="resource-list">
                  {items.map((resource) => (
                    <article key={resource.url}>
                      <div>
                        <h3>
                          <a href={resource.url} target="_blank" rel="noreferrer">
                            {resource.title} <span aria-hidden="true">↗</span>
                          </a>
                        </h3>
                        <p>{resource.description}</p>
                      </div>
                      <ul aria-label={`${resource.title} 标签`}>
                        {resource.tags.map((tag) => <li key={tag}>{tag}</li>)}
                      </ul>
                    </article>
                  ))}
                </div>
              </section>
            );
          })}
        </div>
      </section>

      <footer>
        <p>发现值得收录的资源？在仓库中编辑 <code>data/resources.json</code>。</p>
        <a href="https://github.com/luca-888/ml-portal" target="_blank" rel="noreferrer">提交资源 ↗</a>
      </footer>
    </main>
  );
}
