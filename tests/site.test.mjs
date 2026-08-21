import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const html = await readFile(new URL("../dist/index.html", import.meta.url), "utf8");

test("generates the complete resource catalog", () => {
  assert.match(html, /<title>Gradient Atlas · 梯度图谱<\/title>/);
  assert.match(html, /26 RESOURCES · 5 CATEGORIES/);
  assert.match(html, /PyTorch/);
  assert.match(html, /Hugging Face/);
  assert.match(html, /data\/resources\.json/);
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
    access(new URL("../dist/favicon.svg", import.meta.url)),
    access(new URL("../dist/og.png", import.meta.url)),
    access(new URL("../dist/.nojekyll", import.meta.url)),
  ]);
});
