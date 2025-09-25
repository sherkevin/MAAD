### sound-anomaly 项目服务器准备说明（给 course/管理员）

本说明用于在我们正式上传项目包之前，先完成服务器侧的清理与准备，避免历史环境冲突与网络问题，确保项目到站即跑。

---

#### TL;DR（清单）
- 目录与空间：为我方创建干净工作目录（示例 `/data/<user>/sound-anomaly/`），可写；预留 50–100GB 可用空间。
- 环境冲突：暂时禁用/注释登录脚本中的 `pyenv init` 与多处 PATH 注入；`conda` 仅按需激活，`conda config --set auto_activate_base false`。
- GPU/驱动：`nvidia-smi` 正常；驱动建议 ≥525（PyTorch cu121 轮子最稳）。
- 系统依赖：安装 `build-essential`/`gcc g++ make`、`git`、`curl wget`、`tmux`、`unzip zip`、`libgl1 libglib2.0-0`、`ffmpeg`。
- 网络与代理：保持梯子/VPN 全时在线；放行以下域名 443：`pypi.org`、`files.pythonhosted.org`、`download.pytorch.org`、`github.com`、`raw.githubusercontent.com`、（可选）`huggingface.co`。
- 账户层面持久代理：在用户环境中固化 `http_proxy/https_proxy/no_proxy`，保证非交互进程可用；Cursor/VScode 远程流量放行。

---

#### 目录与清理
- 在我的家目录或数据盘创建：`/data/<user>/sound-anomaly/`（可写）。
- 备份/清理历史无关内容，避免 `conda` 与 `pyenv` 的旧环境残留干扰本次部署。
- 可选：配置公共 pip/torch 轮子缓存目录（如 `/data/pip-cache/`），提升多人安装速度。

#### 环境冲突规避（conda vs pyenv）
- 在 `~/.bashrc`、`~/.zshrc` 中暂时注释 `pyenv init` 与多处 PATH 注入，避免覆盖 `conda`。
- 执行：`conda config --set auto_activate_base false`，防止登录即激活 base。
- 我们将使用项目私有 env（上传后提供一键脚本），不污染系统 Python。

#### 网络与代理（关键）
- 持续保持代理在线，保障 PyPI/PyTorch CDN/GitHub 的稳定连通。
- 如需系统级代理，建议在 `/etc/environment` 或用户 profile 持久化：
  ```bash
  export http_proxy=http://<proxy>
  export https_proxy=http://<proxy>
  export no_proxy=localhost,127.0.0.1,::1,<intranet-subnets>
  ```
- 请确保远程开发工具（Cursor/VSCode）相关流量不被拦截或中途断流。

#### 我们上传后将执行（参考）
- 上传包解压至：`/data/<user>/sound-anomaly/`。
- 进入项目根目录，执行一键环境脚本（脚本随项目提供）：
  ```bash
  cd sound-anomaly-main
  bash scripts/server_setup.sh --env tsad310-cuda --python 3.10 --cuda auto
  ```
  - 无 conda 时：`bash scripts/server_setup.sh --no-conda`
  - 脚本会按服务器 GPU 自动选择 PyTorch CUDA 轮子（优先 cu121）。

#### 数据布局（示例）
- 建议数据根：`/data/datasets/`
- MSL（示例）：`/data/datasets/MSL/{MSL_train.npy, MSL_test.npy, MSL_test_label.npy}`

#### 首次运行（示例）
```bash
conda activate tsad310-cuda   # 或 source .venv-*
cd sound-anomaly-main
python main.py experiment.root='/data/datasets' data.dataset='MSL' \
  model.pre_trained=true model.fusion_type='mlp' model.physics_enable=false \
  model.drift_enable=true model.drift_windows=4 trainer.max_epochs=10
```
- 可选 LLM 融合（需代理稳定）：
  ```bash
  export DASHSCOPE_API_KEY=your_key
  # 然后可切换 fusion_type='hybrid'
  ```

#### 时间评估（首次搭建）
- conda/venv 创建：5–10 分钟
- PyTorch 轮子下载与安装：2–10 分钟（取决于网络）
- 其余依赖与内嵌包安装：5–10 分钟
- 合计：15–35 分钟；若网络慢或首次拉大轮子，可能 30–60 分钟

#### 常见问题
- Conda 与 pyenv 冲突：请确保仅启用一种环境管理工具；避免多重 PATH 注入。
- CUDA/驱动不匹配：优先使用带 CUDA 的官方 PyTorch 轮子；无需系统级 CUDA Toolkit。
- 代理仅在交互端有效：请在非交互/后台进程也继承 `http_proxy/https_proxy`。

#### 联系方式
- 如需配合排查网络/环境，请告知我可登录时段；我会现场验证脚本与下载连通性。

（本文件为上线前说明；项目压缩包上传后，可按文中命令直接部署与首轮运行。）


