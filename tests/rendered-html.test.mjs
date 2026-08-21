import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the resource catalog", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Gradient Atlas · 梯度图谱<\/title>/i);
  assert.match(html, /A CURATED ML LEARNING LIST/);
  assert.match(html, /机器学习，/);
  assert.match(html, /网站会根据清单自动生成/);
  assert.match(html, /PyTorch/);
  assert.match(html, /Hugging Face/);
  assert.match(html, /data\/resources\.json/);
  assert.doesNotMatch(html, /添加收藏|稍后阅读|localStorage/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape/);
});

test("resource data is valid and has unique links", async () => {
  const file = new URL("../data/resources.json", import.meta.url);
  const resources = JSON.parse(await readFile(file, "utf8"));
  const categories = new Set(["repo", "course", "research", "blog", "tool"]);
  const urls = new Set();

  assert.ok(resources.length > 0);
  for (const resource of resources) {
    assert.ok(categories.has(resource.category));
    assert.ok(resource.title.trim());
    assert.ok(resource.description.trim());
    assert.ok(Array.isArray(resource.tags));
    assert.match(resource.url, /^https?:\/\//);
    assert.ok(!urls.has(resource.url), `duplicate URL: ${resource.url}`);
    urls.add(resource.url);
  }
});

test("ships site-specific social metadata", async () => {
  const response = await render();
  const html = await response.text();

  assert.match(html, /property="og:title" content="Gradient Atlas · 梯度图谱"/);
  assert.match(html, /property="og:image" content="https:\/\/raw\.githubusercontent\.com\/luca-888\/ml-portal\/main\/public\/og\.png"/);
  assert.match(html, /name="twitter:card" content="summary_large_image"/);
});
