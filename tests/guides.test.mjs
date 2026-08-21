import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const expectedSlugs = [
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
  "liger-kernel",
];

const root = new URL("../", import.meta.url);
const home = await readFile(new URL("../dist/index.html", import.meta.url), "utf8");
const hub = await readFile(new URL("../dist/guides/index.html", import.meta.url), "utf8");

test("builds every project guide and links it from the directory", () => {
  assert.equal(home.match(/class="guide-badge"/g)?.length, expectedSlugs.length);
  assert.equal(hub.match(/class="guide-index-card"/g)?.length, expectedSlugs.length);
});

for (const slug of expectedSlugs) {
  test(`builds an interactive ${slug} guide`, async () => {
    const source = new URL(`../guides/${slug}/`, import.meta.url);
    const built = new URL(`../dist/guides/${slug}/`, import.meta.url);
    const [html, labJs, labCss] = await Promise.all([
      readFile(new URL("index.html", built), "utf8"),
      readFile(new URL("lab.js", source), "utf8"),
      readFile(new URL("lab.css", source), "utf8"),
    ]);

    assert.match(html, new RegExp(`data-guide="${slug}"`));
    assert.match(html, /id="architecture"/);
    assert.match(html, /id="lab"/);
    assert.match(html, /aria-live=/);
    assert.ok((html.match(/data-architecture-node=/g) ?? []).length >= 5);
    assert.ok((html.match(/class="code-card"/g) ?? []).length >= 2);
    assert.doesNotThrow(() => new Function(labJs));
    assert.match(labCss, new RegExp(`\\[data-guide="${slug.replaceAll("-", "\\-")}"\\]`));

    await Promise.all([
      access(new URL("content.mjs", source)),
      access(new URL("lab.js", built)),
      access(new URL("lab.css", built)),
    ]);
  });
}

test("keeps the guide hub deployable", async () => {
  await Promise.all([
    access(new URL("dist/guides/guide.css", root)),
    access(new URL("dist/guides/guide.js", root)),
  ]);
  assert.match(hub, /项目解读/);
});
