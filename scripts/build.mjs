import { copyFile, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const output = join(root, "dist");
const resources = JSON.parse(
  await readFile(join(root, "data", "resources.json"), "utf8"),
);

const categories = [
  { key: "repo", label: "开源仓库", number: "01" },
  { key: "course", label: "课程", number: "02" },
  { key: "research", label: "论文与研究", number: "03" },
  { key: "blog", label: "博客", number: "04" },
  { key: "tool", label: "工具", number: "05" },
];

validateResources(resources);

const categoryLinks = categories
  .map(
    ({ key, label, number }) => `
      <a href="#${key}"><span>${number}</span>${label}</a>`,
  )
  .join("");

const sections = categories
  .map(({ key, label, number }) => {
    const items = resources.filter((resource) => resource.category === key);
    const rows = items
      .map(
        (resource) => `
          <article>
            <div>
              <h3><a href="${escapeHtml(resource.url)}" target="_blank" rel="noreferrer">${escapeHtml(resource.title)} <span aria-hidden="true">↗</span></a></h3>
              <p>${escapeHtml(resource.description)}</p>
            </div>
            <ul aria-label="${escapeHtml(resource.title)} 标签">
              ${resource.tags.map((tag) => `<li>${escapeHtml(tag)}</li>`).join("")}
            </ul>
          </article>`,
      )
      .join("");

    return `
      <section class="resource-section" id="${key}">
        <div class="section-heading">
          <span>${number}</span>
          <h2>${label}</h2>
          <em>${items.length}</em>
        </div>
        <div class="resource-list">${rows}</div>
      </section>`;
  })
  .join("");

const html = `<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Gradient Atlas · 梯度图谱</title>
    <meta name="description" content="从仓库资源清单自动生成的机器学习学习导航。">
    <link rel="canonical" href="https://luca-888.github.io/ml-portal/">
    <link rel="icon" href="./favicon.svg" type="image/svg+xml">
    <link rel="stylesheet" href="./styles.css">
    <meta property="og:type" content="website">
    <meta property="og:title" content="Gradient Atlas · 梯度图谱">
    <meta property="og:description" content="一份简单、持续维护的机器学习学习资源清单。">
    <meta property="og:url" content="https://luca-888.github.io/ml-portal/">
    <meta property="og:image" content="https://luca-888.github.io/ml-portal/og.png">
    <meta name="twitter:card" content="summary_large_image">
  </head>
  <body>
    <main>
      <header class="site-header">
        <a class="brand" href="#top">Gradient Atlas</a>
        <nav aria-label="页面导航">
          <a href="#resources">资源</a>
          <a href="https://github.com/luca-888/ml-portal" target="_blank" rel="noreferrer">GitHub ↗</a>
        </nav>
      </header>

      <section class="hero" id="top">
        <p class="eyebrow">A CURATED ML LEARNING LIST</p>
        <h1>机器学习，<br>从这些资源开始。</h1>
        <div class="hero-meta">
          <p>一份简单、持续维护的 ML 学习资源清单。仓库是唯一数据源，网站会根据清单自动生成。</p>
          <span>${resources.length} RESOURCES · ${categories.length} CATEGORIES</span>
        </div>
      </section>

      <section class="catalog" id="resources">
        <aside>
          <p>按类别浏览</p>
          <nav aria-label="资源分类">${categoryLinks}</nav>
        </aside>
        <div class="resource-sections">${sections}</div>
      </section>

      <footer>
        <p>发现值得收录的资源？在仓库中编辑 <code>data/resources.json</code>。</p>
        <a href="https://github.com/luca-888/ml-portal" target="_blank" rel="noreferrer">提交资源 ↗</a>
      </footer>
    </main>
  </body>
</html>`;

await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
await writeFile(join(output, "index.html"), html);
await copyFile(join(root, "site", "styles.css"), join(output, "styles.css"));
await copyFile(join(root, "public", "favicon.svg"), join(output, "favicon.svg"));
await copyFile(join(root, "public", "og.png"), join(output, "og.png"));
await writeFile(join(output, ".nojekyll"), "");

console.log(`Built ${resources.length} resources into dist/index.html`);

function validateResources(items) {
  const allowedCategories = new Set(categories.map(({ key }) => key));
  const urls = new Set();

  if (!Array.isArray(items) || items.length === 0) {
    throw new Error("data/resources.json must contain at least one resource");
  }

  for (const [index, resource] of items.entries()) {
    if (!allowedCategories.has(resource.category)) {
      throw new Error(`Resource ${index + 1} has an unknown category`);
    }
    if (!resource.title?.trim() || !resource.description?.trim()) {
      throw new Error(`Resource ${index + 1} is missing a title or description`);
    }
    if (!Array.isArray(resource.tags)) {
      throw new Error(`Resource ${index + 1} must have a tags array`);
    }

    const url = new URL(resource.url);
    if (!new Set(["http:", "https:"]).has(url.protocol)) {
      throw new Error(`Resource ${index + 1} must use an HTTP(S) URL`);
    }
    if (urls.has(resource.url)) {
      throw new Error(`Duplicate resource URL: ${resource.url}`);
    }
    urls.add(resource.url);
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
