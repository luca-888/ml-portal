# Gradient Atlas · 梯度图谱

一份简单、持续维护的机器学习学习资源清单，以及一个由这份清单自动生成的网站。

## 收录内容

- 开源仓库
- 课程
- 论文与研究入口
- 博客
- 工具

所有资源都保存在 [`data/resources.json`](data/resources.json)。网站会读取这一个文件并按分类生成页面，没有数据库、登录或后台管理。

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

提交后，网站会在下一次构建时自动更新。

## 本地运行

需要 Node.js `>=22.13.0`。

```bash
npm install
npm run dev
```

验证改动：

```bash
npm test
```

## License

MIT
