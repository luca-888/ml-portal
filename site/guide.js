const progress = document.querySelector(".reading-progress span");
const tocLinks = [...document.querySelectorAll(".guide-toc a[href^='#']")];
const sections = [...document.querySelectorAll("[data-guide-section]")];

function updateProgress() {
  const scrollable = document.documentElement.scrollHeight - innerHeight;
  const ratio = scrollable > 0 ? scrollY / scrollable : 0;
  progress.style.transform = `scaleX(${Math.min(1, Math.max(0, ratio))})`;
}

addEventListener("scroll", updateProgress, { passive: true });
updateProgress();

if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      for (const link of tocLinks) {
        link.toggleAttribute("aria-current", link.hash === `#${visible.target.id}`);
      }
    },
    { rootMargin: "-20% 0px -65%", threshold: [0.05, 0.4] },
  );
  for (const section of sections) observer.observe(section);
}

for (const button of document.querySelectorAll("[data-copy-code]")) {
  button.addEventListener("click", async () => {
    const code = button.closest(".code-card")?.querySelector("code")?.textContent ?? "";
    await navigator.clipboard.writeText(code);
    button.textContent = "已复制";
    setTimeout(() => { button.textContent = "复制"; }, 1200);
  });
}

const architecture = document.querySelector("[data-architecture]");
if (architecture) {
  const nodes = [...architecture.querySelectorAll("[data-architecture-node]")];
  const detail = architecture.querySelector(".architecture-detail");
  const svg = architecture.querySelector(".architecture-links");

  for (const node of nodes) {
    node.addEventListener("click", () => {
      for (const item of nodes) item.setAttribute("aria-pressed", String(item === node));
      detail.querySelector("strong").textContent = node.querySelector("strong").textContent;
      detail.querySelector("p").textContent = node.querySelector("small").textContent;
    });
  }

  function drawEdges() {
    const container = architecture.querySelector(".architecture-nodes");
    const bounds = container.getBoundingClientRect();
    const architectureBounds = architecture.getBoundingClientRect();
    svg.style.left = `${bounds.left - architectureBounds.left}px`;
    svg.style.top = `${bounds.top - architectureBounds.top}px`;
    svg.style.width = `${bounds.width}px`;
    svg.style.height = `${bounds.height}px`;
    svg.setAttribute("viewBox", `0 0 ${bounds.width} ${bounds.height}`);
    svg.replaceChildren();
    for (const edge of architecture.querySelectorAll("[data-edge-from]")) {
      const from = architecture.querySelector(`[data-architecture-node="${CSS.escape(edge.dataset.edgeFrom)}"]`);
      const to = architecture.querySelector(`[data-architecture-node="${CSS.escape(edge.dataset.edgeTo)}"]`);
      if (!from || !to) continue;
      const a = from.getBoundingClientRect();
      const b = to.getBoundingClientRect();
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", String(a.left - bounds.left + a.width / 2));
      line.setAttribute("y1", String(a.top - bounds.top + a.height / 2));
      line.setAttribute("x2", String(b.left - bounds.left + b.width / 2));
      line.setAttribute("y2", String(b.top - bounds.top + b.height / 2));
      svg.append(line);
    }
  }

  new ResizeObserver(drawEdges).observe(architecture.querySelector(".architecture-nodes"));
  drawEdges();
}
