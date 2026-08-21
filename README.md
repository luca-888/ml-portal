# Gradient Atlas · 梯度图谱

一份简单、持续维护的机器学习学习资源清单。

网站：[luca-888.github.io/ml-portal](https://luca-888.github.io/ml-portal/)

## 工作方式

```text
data/resources.json
        ↓
静态页面生成
        ↓
GitHub Actions
        ↓
GitHub Pages
```

仓库中的 [`data/resources.json`](data/resources.json) 是唯一数据源。提交到 `main` 后，GitHub Actions 会生成纯静态 HTML，并自动发布到 GitHub Pages。

没有数据库、登录、服务器、React、Cloudflare 或额外托管服务。

## 添加资源

在 `data/resources.json` 中新增一项：

```json
{
  "category": "course",
  "title": "资源名称",
  "description": "为什么值得学习。",
  "url": "https://example.com/",
  "tags": ["Topic", "Free"]
}
```

`category` 可选：`repo`、`course`、`research`、`blog`、`tool`。

## 本地生成

需要 Node.js 22 或更高版本，无需安装依赖。

```bash
npm run build
```

生成结果位于 `dist/`。验证资源数据与页面输出：

```bash
npm test
```

## License

MIT
