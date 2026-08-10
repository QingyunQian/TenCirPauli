# TenCirPauli 发布流程

状态：版本无关的发布规则。GitHub 仓库为 `tensorcircuit/TenCirPauli`，PyPI 项目为 `tencirpauli`。每次发布只更新版本元数据、CHANGELOG 和对应的 Git tag，不修改本文档中的流程规则。

## 两类工作流

`.github/workflows/ci.yml` 是日常 CI。每次 push 和 pull request 都会触发，但它只检查代码，不发布任何内容。Rust job 在 Ubuntu 上运行 rustfmt、Clippy 和 workspace tests；Python job 在 Linux、macOS、Windows 以及 Python 3.9、3.13 上用 release maturin 构建并安装扩展，然后运行 pytest。这里的编译是安装/测试步骤，不会把 wheel 上传为 artifact，更不会上传 PyPI。

`.github/workflows/release.yml` 是发行工作流。手动触发 `workflow_dispatch` 时只构建并保存 wheel/sdist artifacts，用于发布前检查；只有 GitHub Release 的 `published` 事件才会执行 PyPI publish job。因此普通 commit、push 和 pull request 不会触发 PyPI 发布。

## 一次性配置

本项目使用 PyPI Trusted Publishing，不保存长期 API token。GitHub 仓库的 environment 名为 `pypi`；PyPI pending publisher 的配置是 owner `tensorcircuit`、repository `TenCirPauli`、workflow `release.yml`、environment `pypi`。这个匹配关系必须保持不变。

GitHub Actions 的 publish job 只有 `id-token: write` 权限，`pypa/gh-action-pypi-publish` 会用短期 OIDC identity 换取上传权限，不需要配置 `PYPI_API_TOKEN`。仓库默认权限应保持 read-only；发布 job 单独申请 OIDC 权限。

## 发布前检查

1. 选定目标版本 `X.Y.Z`，同步 `pyproject.toml`、workspace `Cargo.toml`、`Cargo.lock` 和构建出的 `tencirpauli.__version__`；四者必须完全一致。
2. 在 `CHANGELOG.md` 顶部增加目标版本条目，检查 README、quickstart、API 文档和 examples 没有引用已删除或过时的 public API。
3. 运行 `python scripts/check.py --benchmark skip`，再运行 `mkdocs build --strict`；后者会从当前 wheel 检查 GitHub Pages 的 API 文档。
4. 在干净工作树上构建并检查 wheel 和 sdist，确认包名、版本、许可证文件、依赖和平台产物符合预期。
5. 等待目标 commit 的日常 CI 通过，再创建并推送对应的 `vX.Y.Z` tag。

## 正式发布步骤

1. 在 GitHub 中基于 `vX.Y.Z` tag 创建 Release；点击 Publish release 后才触发发行工作流。手动运行 `workflow_dispatch` 只构建 artifacts，不会上传 PyPI。
2. 工作流分别构建 Linux x86_64/aarch64、macOS x86_64/aarch64、Windows x64 wheel 和一个 sdist，并在上传前检查版本、metadata 和许可证文件；全部成功后才发布到 PyPI。
3. GitHub Pages 工作流在 `main` push 后用当前源码构建 release wheel，再运行 `mkdocs build --strict` 并部署站点；因此 API 页跟随当前代码，而不是跟随 PyPI 上一次发布的 wheel。
4. 发布后检查 PyPI 文件列表，并在干净环境中执行一次 `pip install tencirpauli`、`import tencirpauli` 和 `print(tencirpauli.__version__)` 验证。

PyPI 不允许覆盖同一版本文件，因此版本号和 tag 应视为不可变。若某次发布有问题，应修复后增加版本号，而不是重传相同版本。
