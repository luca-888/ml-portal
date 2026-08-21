const searchInput = document.querySelector("#resource-search");
const filterButtons = [...document.querySelectorAll("[data-filter]")];
const resources = [...document.querySelectorAll("[data-resource]")];
const visibleCount = document.querySelector("#visible-count");
const emptyState = document.querySelector("#empty-state");

let activeCategory = "all";

function applyFilters() {
  const query = searchInput.value.trim().toLowerCase();
  let count = 0;

  for (const resource of resources) {
    const matchesCategory =
      activeCategory === "all" || resource.dataset.category === activeCategory;
    const matchesQuery = resource.dataset.search.includes(query);
    const visible = matchesCategory && matchesQuery;

    resource.hidden = !visible;
    if (visible) count += 1;
  }

  visibleCount.textContent = String(count);
  emptyState.hidden = count !== 0;
}

for (const button of filterButtons) {
  button.addEventListener("click", () => {
    activeCategory = button.dataset.filter;
    for (const item of filterButtons) {
      item.setAttribute("aria-pressed", String(item === button));
    }
    applyFilters();
  });
}

searchInput.addEventListener("input", applyFilters);

document.addEventListener("keydown", (event) => {
  const isTyping = ["INPUT", "TEXTAREA", "SELECT"].includes(
    document.activeElement?.tagName,
  );

  if (event.key === "/" && !isTyping) {
    event.preventDefault();
    searchInput.focus();
  }

  if (event.key === "Escape" && document.activeElement === searchInput) {
    searchInput.value = "";
    applyFilters();
    searchInput.blur();
  }
});
