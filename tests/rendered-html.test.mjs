import assert from "node:assert/strict";
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

test("server-renders the Gradient Atlas portal", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Gradient Atlas · 梯度图谱<\/title>/i);
  assert.match(html, /A PERSONAL MACHINE LEARNING INDEX/);
  assert.match(html, /把好奇心，整理成一张/);
  assert.match(html, /PyTorch/);
  assert.match(html, /Hugging Face/);
  assert.match(html, /添加收藏/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape/);
});

test("ships site-specific social metadata", async () => {
  const response = await render();
  const html = await response.text();

  assert.match(html, /property="og:title" content="Gradient Atlas · 梯度图谱"/);
  assert.match(html, /property="og:image" content="https:\/\/raw\.githubusercontent\.com\/luca-888\/ml-portal\/main\/public\/og\.png"/);
  assert.match(html, /name="twitter:card" content="summary_large_image"/);
});
