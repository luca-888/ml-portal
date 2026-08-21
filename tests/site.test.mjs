import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const html = await readFile(new URL("../dist/index.html", import.meta.url), "utf8");
const resources = JSON.parse(
  await readFile(new URL("../data/resources.json", import.meta.url), "utf8"),
);

test("generates the complete resource catalog", () => {
  assert.match(html, /<title>Gradient Atlas · 梯度图谱<\/title>/);
  assert.match(html, new RegExp(`${resources.length} 个资源`));
  assert.match(html, /PyTorch/);
  assert.match(html, /SGLang/);
  assert.match(html, /Nano-vLLM/);
  assert.match(html, /Hugging Face/);
  assert.match(html, /data\/resources\.json/);
  assert.match(html, /id="resource-search"/);
  assert.match(html, /data-filter="course"/);
  assert.match(html, /data-filter="implementation"/);
  assert.match(html, /class="resource-row"/);
  assert.match(html, /src="https:\/\/github\.com\/pytorch\.png\?size=64"/);
  assert.match(html, /src="https:\/\/github\.com\/fastai\.png\?size=64"/);
  assert.equal(
    html.match(/<img src=/g)?.length,
    resources.filter((resource) => resource.icon).length,
  );
});

test("uses GitHub Pages metadata and relative assets", () => {
  assert.match(html, /https:\/\/luca-888\.github\.io\/ml-portal\//);
  assert.match(html, /href="\.\/styles\.css"/);
  assert.match(html, /href="\.\/favicon\.svg"/);
  assert.doesNotMatch(html, /chatgpt\.site|Cloudflare|codex-preview/);
});

test("produces all deployable static files", async () => {
  await Promise.all([
    access(new URL("../dist/index.html", import.meta.url)),
    access(new URL("../dist/styles.css", import.meta.url)),
    access(new URL("../dist/app.js", import.meta.url)),
    access(new URL("../dist/favicon.svg", import.meta.url)),
    access(new URL("../dist/og.png", import.meta.url)),
    access(new URL("../dist/.nojekyll", import.meta.url)),
  ]);
});
