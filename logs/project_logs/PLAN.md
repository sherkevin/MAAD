## 项目六周发表计划（顶会导向）

### 目标与定位
- 目标会议：ICRA/IROS（机器人时间序列/声音异常）、CVPR/ICCV（多模态/表征）、NeurIPS/ICLR/ICML（算法与校准）。
- 六周目标：形成有竞争力的投稿初稿（含完整实验、消融、代码复现），若强基线差距偏大，则优先投顶会Workshop并滚动增强为主会/期刊版本。

### 三大核心创新点（凝练版）
- 1) DGPA-LLM：分解感知门控聚合 + Qwen 动态权重
  - 将 STL 分解得到的 trend/seasonal/residual/original 四路特征，经轻量门控（两层 MLP）与 Qwen（LLM）提供的先验权重（基于窗口统计文本化提示）进行融合，得到自适应表征后再送入 timm 特征与 PatchCore。
  - 贡献：在非平稳时间序列下，利用语言先验注入任务上下文（工况描述、事件注释、历史现象归纳）提升泛化；保持训练/推理轻量（LLM 仅作为权重建议器，可缓存/蒸馏）。

- 2) DACM：漂移感知记忆库（Drift-Aware Coreset Memory）
  - 在 PatchCore 记忆库上加入简单漂移检测（KS/能量距离/PSI），触发分段式 KCenterGreedy 重采样与增量更新，避免分布漂移导致的性能退化。
  - 贡献：对长时在线/跨域部署更鲁棒，工程可用性强，消融直观。

- 3) PICC：物理先验 + 置信校准 + 延迟敏感（Physics-Informed + Conformal Calibration）
  - 物理先验（PINN风格）：对 seasonal/trend 加入二阶平滑/能量守恒等软约束，对 residual 加入白噪声性/谱稀疏正则；支持用户自定义简单 ODE/PDE 片段（以正则项实现）。
  - 校准与延迟：采用 conformal/分位数校准，报告固定 FPR 下的 TPR 与告警延迟分布，形成“可解释、可部署”的阈值。
  - 贡献：从“检出”走向“可靠决策”，更贴合工业/机器人应用评价标准。

备注：若需严格只写“三点”，则将“PINN与校准”合并为第三点（PICC）。

### 技术方案细化与实现要点（结合论文启发）
- DGPA-LLM
  - 数据侧：沿用 STL 分解（项目已有 `data_factory`），窗口统计特征（均值/方差/峰度/ADF等）文本化 prompt 给 Qwen；Qwen 返回四通道权重建议；门控模块融合 LLM 权重与可学习权重（凸组合）。
  - 模型侧：在 `anomalibv2/.../da_patchcore/torch_model.py` 的 `gather_features` 前新增 `FusionGate`；支持 `fusion={mlp|llm|hybrid}`；LLM 仅CPU/异步调用，支持缓存与蒸馏为小 MLP。为贴合“训练自由度”叙事，提供 `train_free=true/false`，当为 true 时锁定门控为规则+LLM建议，模拟 VISTA 式免训练范式。
  - 代价控制：默认 `mlp`，`llm` 为可选；服务器上可启用 `qwen`（transformers，本地权重），Mac 上用 `mlp` 或 `llm_cached`。

- DACM
  - 触发条件：滑窗分布检验（KS/ED/PSI），阈值来自验证集或稳健分位数；
  - 更新策略：KCenterGreedy 对新窗口 embedding 重采样，与旧记忆拼接后限长；
  - 日志：记录每次更新的覆盖率/多样性分数与性能变化；加入“触发率-收益曲线”以体现系统层权衡（参考 Nature 文风格的系统指标）。

- PINN 启发增强（跨 DGPA/PICC/DACM 的训练与采样策略）
  - 自适应激活与频域特征：在 DGPA 门控中提供 `fusion_act ∈ {sine, gelu, relu}` 与可选 Fourier features；默认 `sine` 或 `fourier+gelu` 以保留高阶导的平滑性。
  - 课程式权重调度：对物理先验正则 λ_physics 采用从小到大逐步放宽（如每 N 步 ×1.5），先拟合数据后收紧物理约束；DGPA 门控也从“均匀/保守”过渡到“残差导向”。
  - GradNorm 多损平衡：自动平衡检测损失、物理正则与校准损失的梯度量级，避免“只拟合边界/只顾物理”。
  - 残差自适应记忆（PRAC）：以当前预测/物理残差作为加权度量指导记忆采样（难例优先）并结合分层/Latin Hypercube 保证覆盖性，替代单纯距离的 KCenterGreedy。
  - 稳定性与可部署指标：新增阈值/分数的滑窗稳定性（方差/抖动）与在漂移触发下的“触发-收益”曲线，突出系统级可靠性。

- PICC
  - PINN正则：对 trend/seasonal 的二阶导惩罚（平滑），对 residual 的谱能量白噪声化（频域均匀度），可配置权重 λ；
  - 校准：验证集上 conformal 估计阈值，推理输出置信区间；
  - 指标：固定 FPR 下 TPR、AUPR、告警延迟分布、阈值稳定性（跨天/跨域）；报告资源开销（延迟、显存）以增强系统可部署叙事。

### 频域与复值表示（来自 Nature 解读的启发）
- 复值系数规范化：统一用 \(c=|c|e^{j\theta}=\cos\theta+j\sin\theta\)，避免实/虚部相位不一致导致的不稳定；\(|c|\) 视作可学习幅度，\(\theta\) 视作相位。
- 频域线性组合：利用 FFT 的线性性，分解各通道（trend/seasonal/residual/original）频谱后做加权叠加；可在 DGPA 的门控中引入“相位敏感”特征（如主频相位、相位差分）。
- 乘积项等非线性：用卷积定理在频域实现（时域乘积≈频域卷积），在不改主干的前提下，以“附加频域分支”实现（轻量、可关）。
- 并行处理与稳定性：借鉴“多波长/正交通道”思想，确保复值实部/虚部处理的正交性；在实现上采用共享相位 \(\theta\) 的复指数表示，减少冗余自由度。
- 归一化与校准：频域分支的输出与时域主干在幅度域统一归一后再融合，避免数值尺度失衡；最终阈值经分位数/Conformal 校准，提升低 FPR 可用性。

### 数据与基线
- 时间序列：SMD/SMAP/MSL/PSM/SWAT（项目已有接口，先修 KPI/YAHOO 的 `lamda` 小 bug）。
- 声音：MIMII/DCASE（新增 `AudioSegLoader` 或 STFT 支路）。
- 你提供的基线表：`/Users/ginger/Downloads/codding way/VOROUS Results Collection.xlsx`（解析为 pandas DataFrame，统一对齐我们实验指标）。

### 实验协议与可复现性
- 统一窗口/步长/seed，报告均值±方差；
- 指标：AUROC/AUPR、固定 FPR 的 TPR、告警延迟、稳定性；
- 消融：
  - DGPA：单通道/静态/MLP/LLM/Hybrid；
  - DACM：无更新/定期/漂移触发；
  - PINN：无/二阶平滑/能量守恒/组合；
  - 校准：无/分位数/Conformal；
  - 频域：无/时频融合（可作为附加项）。

### 资源与排期（6周）
- 环境：本地Mac调试（MPS/CPU），服务器3090×4 训练评测。
- 周计划
  - 第1周：代码清理与可跑基线（已完成）→ 修 KPI/YAHOO、补 README、脚本化一键跑；阅读两篇论文（项目对应论文、Nature 文）并输出要点与故事线。
  - 第2周：DGPA-LLM 实装（mlp/llm/hybrid 三模式、缓存与蒸馏）、小规模验证与消融；
  - 第3周：DACM 实装（漂移检验、重采样策略、日志），大数据集跑通并记录更新曲线；
  - 第4周：PICC 实装（PINN 正则与 conformal 校准），完善延迟敏感实验；
  - 第5周：大规模实验/消融与可解释性可视化、SOTA 对比、结果表格与图；
  - 第6周：论文撰写与润色（主文+附录+代码整理），若必要进行一轮快速翻修。

### 风险与备选方案（务实）
- LLM 时延或不可用：使用缓存/蒸馏到 MLP；或采用规则库（基于统计阈值）替代。
- PINN 正则不显著：降权并聚焦于校准与DACM；或改为更通用的频域稀疏先验。
- 公开集差异大：引入自适应阈值/归一化（已有 normalize），加权投票聚合窗口分数。
- timm/HF 权重下载受限：本地缓存/提供离线包；或改为随机初始化+冻结前层。

### 代码实施清单（高优先级）
- [ ] 修复 `data_factory/dataloader.py` 中 KPI/YAHOO 的 `lamda` 未定义问题
- [ ] `FusionGate` 与 `fusion={mlp|llm|hybrid}` 配置（Hydra）
- [ ] Qwen 接入与缓存/蒸馏（transformers，本地推理；可选量化）
- [ ] DACM 的漂移检测与记忆更新策略封装
- [ ] PINN 正则项（时间/频域）与权重配置
- [ ] Conformal 校准 Callback 与延迟统计
- [ ] 统一日志/可视化（告警热力图、阈值漂移、记忆更新时刻）
- [ ] 实验脚本与表格导出（对齐 Excel 基线）
 - [ ] GradNorm 多损平衡与课程式 λ_physics 调度
 - [ ] DGPA `fusion_act` 与 Fourier features 选项
 - [ ] 残差自适应记忆 PRAC（残差加权 + 分层覆盖）
 - [ ] 阈值/分数稳定性指标与“触发-收益”曲线输出

### 论文写作结构（故事线）
- 问题动机：非平稳时间序列/声音异常检测在实际场景的挑战（漂移、先验、置信度）。
- 方法概览：DGPA-LLM → DACM → PICC（轻量、端到端、可部署）。
- 实现与复杂度：门控轻量、LLM 可缓存、记忆更新低开销、PINN 正则通用。
- 实验：广泛基线、SOTA 对比、消融、稳定性与延迟、可解释性、开源复现。
- 结论与展望：跨域泛化、在线部署、后续扩展（音频/多模态）。


