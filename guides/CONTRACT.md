# Project guide contract

Each project guide owns exactly one directory under `guides/<slug>/` and must
not edit files outside that directory.

Required files:

- `content.mjs`: Chinese teaching content and source metadata.
- `lab.js`: browser-side interaction scoped to `[data-guide="<slug>"]`.
- `lab.css`: styles scoped below `[data-guide="<slug>"]`.

`content.mjs` must default-export this shape:

```js
export default {
  slug: "project-slug",
  title: "Project name",
  subtitle: "One factual sentence",
  repoUrl: "https://github.com/owner/repo",
  accent: "#hex",
  tags: ["Topic", "Topic"],
  audience: "Who should read this",
  prerequisites: ["..."],
  outcomes: ["..."],
  overview: ["Two or three concise paragraphs."],
  architecture: {
    title: "Diagram title",
    description: "How to read it",
    nodes: [{ id: "short-id", label: "Label", detail: "Explanation" }],
    edges: [{ from: "short-id", to: "short-id", label: "optional" }],
  },
  sections: [
    {
      id: "stable-id",
      title: "Section title",
      summary: "One-sentence summary",
      paragraphs: ["..."],
      bullets: ["..."],
      code: {
        title: "What this excerpt shows",
        language: "python",
        snippet: "A short, verified excerpt or faithful pseudocode",
        sourceUrl: "https://github.com/owner/repo/blob/<ref>/path",
        sourceLabel: "path/in/repo",
        note: "State explicitly when adapted or pseudocode",
      },
      visual: {
        title: "Visual title",
        html: `<div class="project-specific-markup">...</div>`,
        caption: "What changes or what to notice",
      },
    },
  ],
  lab: {
    title: "Interactive lab title",
    intro: "What the controls demonstrate",
    html: `<div class="project-lab">...</div>`,
  },
  takeaways: ["..."],
  sources: [{ label: "Source", url: "https://..." }],
};
```

Content requirements:

- Use Chinese and factual wording. Do not use promotional adjectives.
- Explain mechanisms, tradeoffs, and code paths rather than summarizing README.
- Include 4–6 sections and 2–4 short code excerpts. Keep each excerpt below 20
  lines. Link every excerpt to the exact upstream file. If code is adapted or
  pseudocode, label it clearly.
- Use only official repository/docs/papers as factual sources.
- `architecture` must contain 5–10 nodes and valid edges.
- `lab.html` must expose controls and an output/diagram. `lab.js` must make the
  lab work without dependencies or network access.
- The lab must support keyboard use, update an `aria-live` result, and respect
  `prefers-reduced-motion`.
- `lab.css` must be responsive and all selectors must begin with
  `[data-guide="<slug>"]`.
- Do not add global styles, external fonts, frameworks, packages, build steps,
  images, or generated binaries.
- Use inline SVG only for diagrams that cannot be expressed with HTML/CSS.
- Do not commit or push.
