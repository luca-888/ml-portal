import { copyFile, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { buildGuides, loadGuides } from "./guide-builder.mjs";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const output = join(root, "dist");
const resources = JSON.parse(
  await readFile(join(root, "data", "resources.json"), "utf8"),
);
const guides = await loadGuides(root);
const guideByRepo = new Map(guides.map((guide) => [guide.repoUrl, guide]));

const categories = [
  { key: "repo", label: "开源仓库", number: "01" },
  { key: "implementation", label: "学习实现", number: "02" },
  { key: "course", label: "课程", number: "03" },
  { key: "research", label: "论文与研究", number: "04" },
  { key: "blog", label: "博客", number: "05" },
  { key: "tool", label: "工具", number: "06" },
];

validateResources(resources);

const categoryLabels = new Map(
  categories.map((category) => [category.key, category.label]),
);

const filters = [
  { key: "all", label: "全部", count: resources.length },
  ...categories.map((category) => ({
    key: category.key,
    label: category.label,
    count: resources.filter((resource) => resource.category === category.key).length,
  })),
]
  .map(
    ({ key, label, count }, index) => `
      <button type="button" data-filter="${key}" aria-pressed="${index === 0}">
        ${label}<span>${count}</span>
      </button>`,
  )
  .join("");

const rows = resources
  .map((resource, index) => {
    const categoryLabel = categoryLabels.get(resource.category);
    const iconUrl = resource.icon ?? null;
    const guide = guideByRepo.get(resource.url);
    const searchable = [
      resource.title,
      resource.description,
      categoryLabel,
      ...resource.tags,
    ].join(" ");
    return `
      <article class="resource-row" data-resource data-category="${resource.category}" data-search="${escapeHtml(searchable.toLowerCase())}">
        <span class="row-index">${String(index + 1).padStart(2, "0")}</span>
        <div class="resource-name">
          ${iconUrl ? `<img src="${escapeHtml(iconUrl)}" alt="" width="18" height="18" loading="lazy" decoding="async">` : ""}
          <h2><a href="${escapeHtml(resource.url)}" target="_blank" rel="noreferrer">${escapeHtml(resource.title)}</a></h2>
          ${guide ? `<a class="guide-badge" href="./guides/${escapeHtml(guide.slug)}/">解读</a>` : ""}
        </div>
        <span class="category-label">${escapeHtml(categoryLabel)}</span>
        <p class="resource-description">${escapeHtml(resource.description)}</p>
        <ul aria-label="${escapeHtml(resource.title)} 标签">
          ${resource.tags.map((tag) => `<li>${escapeHtml(tag)}</li>`).join("")}
        </ul>
        <a class="open-link" href="${escapeHtml(resource.url)}" target="_blank" rel="noreferrer" aria-label="打开 ${escapeHtml(resource.title)}">↗</a>
      </article>`;
  })
  .join("");

const html = `<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Gradient Atlas · 梯度图谱</title>
    <meta name="description" content="从仓库资源清单生成的机器学习导航。">
    <link rel="canonical" href="https://luca-888.github.io/ml-portal/">
    <link rel="icon" href="./favicon.svg" type="image/svg+xml">
    <link rel="stylesheet" href="./styles.css">
    <script src="./app.js" defer></script>
    <meta property="og:type" content="website">
    <meta property="og:title" content="Gradient Atlas · 梯度图谱">
    <meta property="og:description" content="机器学习仓库、学习实现、课程、论文、博客与工具。">
    <meta property="og:url" content="https://luca-888.github.io/ml-portal/">
    <meta property="og:image" content="https://luca-888.github.io/ml-portal/og.png">
    <meta name="twitter:card" content="summary_large_image">
  </head>
  <body>
    <main>
      <header class="site-header" id="top">
        <a class="brand" href="#top"><span>G/</span> Gradient Atlas</a>
        <div>
          <span>${resources.length} 个资源</span>
          ${guides.length ? `<a href="./guides/">项目解读 ${guides.length}</a>` : ""}
          <a href="https://github.com/luca-888/ml-portal" target="_blank" rel="noreferrer">GitHub ↗</a>
        </div>
      </header>

      <section class="directory-intro">
        <div>
          <p class="eyebrow">ML LEARNING DIRECTORY</p>
          <h1>机器学习资源索引</h1>
        </div>
        <p>仓库、学习实现、课程、论文、博客与工具。支持搜索和分类筛选。</p>
      </section>

      <section class="directory" id="resources">
        <div class="directory-controls">
          <label class="search-field">
            <span class="sr-only">搜索资源</span>
            <span aria-hidden="true">⌕</span>
            <input id="resource-search" type="search" placeholder="搜索名称、说明或标签…" autocomplete="off">
            <kbd>/</kbd>
          </label>
          <div class="filter-group" aria-label="资源分类">${filters}</div>
          <p class="result-count" aria-live="polite"><strong id="visible-count">${resources.length}</strong> / ${resources.length}</p>
        </div>

        <div class="column-head" aria-hidden="true">
          <span>#</span><span>资源</span><span>分类</span><span>简介</span><span>标签</span><span></span>
        </div>
        <div class="resource-list" id="resource-list">${rows}</div>
        <div class="empty-state" id="empty-state" hidden>
          <strong>没有匹配的资源</strong>
          <span>试试更短的关键词或切换分类。</span>
        </div>
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
await copyFile(join(root, "site", "app.js"), join(output, "app.js"));
await copyFile(join(root, "public", "favicon.svg"), join(output, "favicon.svg"));
await copyFile(join(root, "public", "og.png"), join(output, "og.png"));
await writeFile(join(output, ".nojekyll"), "");
await buildGuides({ root, output, guides });

console.log(`Built ${resources.length} resources and ${guides.length} guides into dist/`);

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

    if (resource.icon) {
      const icon = new URL(resource.icon);
      if (!new Set(["http:", "https:"]).has(icon.protocol)) {
        throw new Error(`Resource ${index + 1} icon must use an HTTP(S) URL`);
      }
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
