import { access, copyFile, mkdir, readdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const guideOrder = [
  "nano-vllm",
  "mini-sglang",
  "picotron",
  "nanochat",
  "llms-from-scratch",
  "llm-c",
  "micrograd",
  "build-nanogpt",
  "gpu-puzzles",
  "minbpe",
];

export async function loadGuides(root) {
  const guidesRoot = join(root, "guides");
  const entries = await readdir(guidesRoot, { withFileTypes: true });
  const guides = [];

  for (const entry of entries) {
    if (!entry.isDirectory() || entry.name.startsWith("_")) continue;
    const directory = join(guidesRoot, entry.name);
    const contentPath = join(directory, "content.mjs");
    try {
      await Promise.all([
        access(contentPath),
        access(join(directory, "lab.js")),
        access(join(directory, "lab.css")),
      ]);
    } catch {
      continue;
    }

    const module = await import(pathToFileURL(contentPath).href);
    validateGuide(module.default, entry.name);
    guides.push({ ...module.default, directory });
  }

  return guides.sort((a, b) => {
    const aIndex = guideOrder.indexOf(a.slug);
    const bIndex = guideOrder.indexOf(b.slug);
    return (aIndex < 0 ? Infinity : aIndex) - (bIndex < 0 ? Infinity : bIndex)
      || a.title.localeCompare(b.title, "en");
  });
}

export async function buildGuides({ root, output, guides }) {
  const guidesOutput = join(output, "guides");
  await mkdir(guidesOutput, { recursive: true });
  await copyFile(join(root, "site", "guide.css"), join(guidesOutput, "guide.css"));
  await copyFile(join(root, "site", "guide.js"), join(guidesOutput, "guide.js"));
  await writeFile(join(guidesOutput, "index.html"), renderGuideIndex(guides));

  for (const [index, guide] of guides.entries()) {
    const target = join(guidesOutput, guide.slug);
    await mkdir(target, { recursive: true });
    const previous = guides[(index - 1 + guides.length) % guides.length];
    const next = guides[(index + 1) % guides.length];
    await writeFile(join(target, "index.html"), renderGuide(guide, previous, next));
    await copyFile(join(guide.directory, "lab.js"), join(target, "lab.js"));
    await copyFile(join(guide.directory, "lab.css"), join(target, "lab.css"));
  }
}

function renderGuide(guide, previous, next) {
  const sectionLinks = guide.sections
    .map((section, index) => `<a href="#${escapeHtml(section.id)}"><span>${String(index + 1).padStart(2, "0")}</span>${escapeHtml(section.title)}</a>`)
    .join("");
  const architectureNodes = guide.architecture.nodes
    .map((node, index) => `<button class="architecture-node" type="button" data-architecture-node="${escapeHtml(node.id)}" style="--node-index:${index}" aria-pressed="${index === 0}"><span>${String(index + 1).padStart(2, "0")}</span><strong>${escapeHtml(node.label)}</strong><small>${escapeHtml(node.detail)}</small></button>`)
    .join("");
  const architectureEdges = guide.architecture.edges
    .map((edge) => `<li data-edge-from="${escapeHtml(edge.from)}" data-edge-to="${escapeHtml(edge.to)}">${escapeHtml(edge.label ?? `${edge.from} → ${edge.to}`)}</li>`)
    .join("");
  const sections = guide.sections.map(renderSection).join("");
  const sources = guide.sources
    .map((source, index) => `<li><span>${String(index + 1).padStart(2, "0")}</span><a href="${escapeHtml(source.url)}" target="_blank" rel="noreferrer">${escapeHtml(source.label)} ↗</a></li>`)
    .join("");
  const tags = guide.tags.map((tag) => `<li>${escapeHtml(tag)}</li>`).join("");
  const prerequisites = guide.prerequisites.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  const outcomes = guide.outcomes.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  const takeaways = guide.takeaways.map((item, index) => `<li><span>${String(index + 1).padStart(2, "0")}</span><p>${escapeHtml(item)}</p></li>`).join("");

  return `<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>${escapeHtml(guide.title)} 解读 · Gradient Atlas</title>
    <meta name="description" content="${escapeHtml(guide.subtitle)}">
    <meta name="theme-color" content="${escapeHtml(guide.accent)}">
    <link rel="icon" href="../../favicon.svg" type="image/svg+xml">
    <link rel="stylesheet" href="../guide.css">
    <link rel="stylesheet" href="./lab.css">
    <script src="../guide.js" defer></script>
    <script src="./lab.js" defer></script>
  </head>
  <body data-guide="${escapeHtml(guide.slug)}" style="--guide-accent:${escapeHtml(guide.accent)}">
    <a class="skip-link" href="#guide-content">跳到正文</a>
    <div class="reading-progress" aria-hidden="true"><span></span></div>
    <header class="guide-header">
      <a class="guide-brand" href="../../"><span>G/</span><strong>Gradient Atlas</strong></a>
      <nav aria-label="页面导航">
        <a href="../">全部解读</a>
        <a href="${escapeHtml(guide.repoUrl)}" target="_blank" rel="noreferrer">GitHub ↗</a>
      </nav>
    </header>

    <main id="guide-content">
      <section class="guide-hero">
        <div class="hero-copy">
          <p class="guide-kicker">PROJECT GUIDE · ${escapeHtml(guide.slug.toUpperCase())}</p>
          <h1>${escapeHtml(guide.title)}</h1>
          <p class="guide-subtitle">${escapeHtml(guide.subtitle)}</p>
          <ul class="guide-tags" aria-label="主题">${tags}</ul>
        </div>
        <div class="hero-meta">
          <div><span>适合谁</span><p>${escapeHtml(guide.audience)}</p></div>
          <div><span>前置知识</span><ul>${prerequisites}</ul></div>
          <div><span>读完能够</span><ul>${outcomes}</ul></div>
        </div>
      </section>

      <div class="guide-layout">
        <aside class="guide-toc">
          <p>本页目录</p>
          <a href="#overview"><span>00</span>概览</a>
          <a href="#architecture"><span>↳</span>系统结构</a>
          ${sectionLinks}
          <a href="#lab"><span>LAB</span>交互实验</a>
          <a href="#takeaways"><span>✓</span>结论</a>
        </aside>

        <div class="guide-article">
          <section class="guide-section guide-overview" id="overview" data-guide-section>
            <div class="section-heading"><span>00</span><div><p>OVERVIEW</p><h2>先建立整体模型</h2></div></div>
            <div class="overview-grid">${guide.overview.map((paragraph) => `<p>${escapeHtml(paragraph)}</p>`).join("")}</div>
          </section>

          <section class="guide-section" id="architecture" data-guide-section>
            <div class="section-heading"><span>↳</span><div><p>ARCHITECTURE</p><h2>${escapeHtml(guide.architecture.title)}</h2></div></div>
            <p class="section-summary">${escapeHtml(guide.architecture.description)}</p>
            <div class="architecture" data-architecture>
              <svg class="architecture-links" aria-hidden="true"></svg>
              <div class="architecture-nodes">${architectureNodes}</div>
              <div class="architecture-detail" aria-live="polite"><span>当前节点</span><strong>${escapeHtml(guide.architecture.nodes[0].label)}</strong><p>${escapeHtml(guide.architecture.nodes[0].detail)}</p></div>
              <ol class="architecture-edge-data" hidden>${architectureEdges}</ol>
            </div>
          </section>

          ${sections}

          <section class="guide-section guide-lab" id="lab" data-guide-section>
            <div class="section-heading"><span>LAB</span><div><p>INTERACTIVE</p><h2>${escapeHtml(guide.lab.title)}</h2></div></div>
            <p class="section-summary">${escapeHtml(guide.lab.intro)}</p>
            <div class="lab-frame">${guide.lab.html}</div>
          </section>

          <section class="guide-section" id="takeaways" data-guide-section>
            <div class="section-heading"><span>✓</span><div><p>TAKEAWAYS</p><h2>保留这些判断</h2></div></div>
            <ol class="takeaway-list">${takeaways}</ol>
          </section>

          <section class="guide-sources" aria-labelledby="sources-title">
            <h2 id="sources-title">来源</h2>
            <ol>${sources}</ol>
          </section>
        </div>
      </div>

      <nav class="guide-pagination" aria-label="项目解读翻页">
        <a href="../${escapeHtml(previous.slug)}/"><span>上一篇</span><strong>← ${escapeHtml(previous.title)}</strong></a>
        <a href="../${escapeHtml(next.slug)}/"><span>下一篇</span><strong>${escapeHtml(next.title)} →</strong></a>
      </nav>
    </main>
  </body>
</html>`;
}

function renderGuideIndex(guides) {
  const cards = guides.map((guide, index) => `<article class="guide-index-card" style="--card-accent:${escapeHtml(guide.accent)}">
    <a href="./${escapeHtml(guide.slug)}/">
      <span>${String(index + 1).padStart(2, "0")}</span>
      <div><p>${escapeHtml(guide.tags.join(" · "))}</p><h2>${escapeHtml(guide.title)}</h2><p>${escapeHtml(guide.subtitle)}</p></div>
      <strong>进入解读 →</strong>
    </a>
  </article>`).join("");

  return `<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>项目解读 · Gradient Atlas</title>
    <meta name="description" content="机器学习项目机制、源码与交互实验。">
    <link rel="icon" href="../favicon.svg" type="image/svg+xml">
    <link rel="stylesheet" href="./guide.css">
  </head>
  <body>
    <header class="guide-header">
      <a class="guide-brand" href="../"><span>G/</span><strong>Gradient Atlas</strong></a>
      <nav><a href="../">资源索引</a><a href="https://github.com/luca-888/ml-portal" target="_blank" rel="noreferrer">GitHub ↗</a></nav>
    </header>
    <main class="guide-index">
      <section class="guide-index-hero">
        <p>PROJECT GUIDES</p><h1>项目解读</h1>
        <div><strong>${guides.length}</strong><span>个项目 · 机制图 · 关键源码 · 交互实验</span></div>
      </section>
      <section class="guide-index-grid" aria-label="项目解读列表">${cards || "<p>内容生成中。</p>"}</section>
    </main>
  </body>
</html>`;
}

function renderSection(section, index) {
  const code = section.code
    ? `<div class="code-card"><div class="code-toolbar"><div><span>${escapeHtml(section.code.language)}</span><strong>${escapeHtml(section.code.title)}</strong></div><button type="button" data-copy-code>复制</button></div><pre><code>${escapeHtml(section.code.snippet)}</code></pre><div class="code-source"><p>${escapeHtml(section.code.note ?? "")}</p><a href="${escapeHtml(section.code.sourceUrl)}" target="_blank" rel="noreferrer">${escapeHtml(section.code.sourceLabel)} ↗</a></div></div>`
    : "";
  const visual = section.visual
    ? `<figure class="section-visual"><div class="visual-heading"><strong>${escapeHtml(section.visual.title)}</strong><span>VISUAL</span></div>${section.visual.html}<figcaption>${escapeHtml(section.visual.caption)}</figcaption></figure>`
    : "";
  const bullets = section.bullets?.length
    ? `<ul class="section-bullets">${section.bullets.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
    : "";

  return `<section class="guide-section" id="${escapeHtml(section.id)}" data-guide-section>
    <div class="section-heading"><span>${String(index + 1).padStart(2, "0")}</span><div><p>MECHANISM</p><h2>${escapeHtml(section.title)}</h2></div></div>
    <p class="section-summary">${escapeHtml(section.summary)}</p>
    <div class="section-prose">${section.paragraphs.map((paragraph) => `<p>${escapeHtml(paragraph)}</p>`).join("")}</div>
    ${bullets}${visual}${code}
  </section>`;
}

function validateGuide(guide, directoryName) {
  if (!guide || guide.slug !== directoryName) throw new Error(`Guide ${directoryName} has an invalid slug`);
  for (const field of ["title", "subtitle", "repoUrl", "audience"]) {
    if (!guide[field]?.trim()) throw new Error(`Guide ${guide.slug} is missing ${field}`);
  }
  if (!/^#[0-9a-f]{6}$/i.test(guide.accent)) throw new Error(`Guide ${guide.slug} has an invalid accent`);
  if (!Array.isArray(guide.sections) || guide.sections.length < 4) throw new Error(`Guide ${guide.slug} needs at least four sections`);
  if (!guide.architecture?.nodes?.length || !guide.architecture?.edges?.length) throw new Error(`Guide ${guide.slug} needs an architecture graph`);
  const nodeIds = new Set(guide.architecture.nodes.map((node) => node.id));
  for (const edge of guide.architecture.edges) {
    if (!nodeIds.has(edge.from) || !nodeIds.has(edge.to)) throw new Error(`Guide ${guide.slug} has an invalid architecture edge`);
  }
  if (!guide.lab?.html?.trim()) throw new Error(`Guide ${guide.slug} needs an interactive lab`);
  if (!guide.sources?.length) throw new Error(`Guide ${guide.slug} needs sources`);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
