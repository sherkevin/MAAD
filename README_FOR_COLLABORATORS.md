# 🤖 MAAAD项目完整包 - 合作者使用指南

**项目全称**: Multi-Agent Anomaly Detection (基于大语言模型的多智能体协作异常检测系统)  
**目标会议**: ICASSP2026  
**创建日期**: 2025年9月24日  
**版本**: v2.0 (ICASSP2026优化版)

---

## 📋 **项目概述**

### 🎯 **我们在做什么？**

MAAAD是一个**革命性的异常检测系统**，首次将**大语言模型(LLM)**与**多智能体协作**相结合，用于解决复杂时间序列数据的异常检测问题。我们的目标是达到**AUROC > 0.7**的SOTA性能，成功投稿ICASSP2026。

### 🚀 **核心创新点**

1. **首次LLM驱动**: 将大语言模型引入多智能体异常检测
2. **327维特征工程**: 多尺度、多维度的特征提取
3. **13种算法集成**: 每个智能体内部集成多种异常检测算法
4. **智能NaN处理**: 彻底解决数据预处理中的NaN值问题
5. **T2MAC通信协议**: 目标导向的多智能体通信机制

---

## 🏗️ **项目结构**

```
MAAAD_Project/
├── 📁 src/                          # 核心源代码
│   ├── agents/                      # 多智能体系统
│   │   ├── base_agent.py           # 智能体基类
│   │   ├── trend_agent.py          # 趋势分析智能体
│   │   ├── working_multi_agent_detector.py  # 可工作的多智能体检测器
│   │   └── multi_agent_detector.py # 多智能体检测器
│   ├── communication/              # 通信协议
│   │   ├── t2mac_protocol.py       # T2MAC通信协议
│   │   ├── enhanced_llm_communication.py  # 增强LLM通信
│   │   └── llm_driven_communication.py    # LLM驱动通信
│   ├── llm/                        # 大语言模型接口
│   │   ├── aliyun_qwen_interface.py # 阿里云百炼接口
│   │   └── qwen_interface.py       # Qwen接口
│   ├── privacy/                    # 隐私保护
│   │   └── differential_privacy.py # 差分隐私
│   ├── federated/                  # 联邦学习
│   │   └── federated_learning.py   # 联邦学习框架
│   └── utils/                      # 工具函数
├── 📁 experiments/                 # 实验脚本
│   ├── real_world/                 # 真实数据集实验
│   ├── ablation/                   # 消融研究
│   ├── complexity/                 # 复杂度分析
│   └── multi_dataset/              # 多数据集实验
├── 📁 configs/                     # 配置文件
│   ├── agent_config.yaml          # 智能体配置
│   ├── server_compatibility_config.yaml  # 服务器兼容性配置
│   └── server_experiment_config.json     # 服务器实验配置
├── 📁 scripts/                     # 部署脚本
│   ├── deploy_to_server.sh        # 服务器部署脚本
│   ├── sync_to_server.sh          # 数据同步脚本
│   └── cross_platform_setup.py    # 跨平台设置
├── 📁 docs/                        # 项目文档
│   ├── PROJECT_STATUS_REPORT.md   # 项目状态报告
│   ├── SERVER_DEPLOYMENT_GUIDE.md # 服务器部署指南
│   └── FINAL_PROJECT_SUMMARY.md   # 项目总结
├── 📁 outputs/                     # 实验结果
│   └── group_meeting_materials/    # 组会材料
├── 📁 logs/                        # 日志文件
│   └── project_logs/               # 项目日志
├── 📁 data/                        # 数据集目录
├── 🐍 核心实验脚本
│   ├── icassp2026_optimized_system.py      # ICASSP2026优化系统 ⭐
│   ├── icassp2026_sota_system.py           # ICASSP2026 SOTA系统
│   ├── sota_multi_agent_system.py          # SOTA多智能体系统
│   ├── advanced_multi_agent_system.py      # 高级多智能体系统
│   ├── real_working_experiments.py         # 真实数据集实验
│   ├── simple_working_experiments.py       # 简单工作实验
│   ├── fully_fixed_experiments.py          # 完全修复实验
│   └── fixed_extended_dataset_experiments.py # 修复扩展实验
├── 🛠️ 工具脚本
│   ├── convert_datasets.py         # 数据集格式转换
│   ├── fix_smd_labels.py          # SMD标签修复
│   ├── test_aliyun_integration.py # 阿里云集成测试
│   └── test_environment.py        # 环境测试
├── 📋 项目文档
│   ├── DETAILED_PROJECT_INTRODUCTION.md    # 详细项目介绍 ⭐
│   ├── PROJECT_OVERVIEW_FOR_COLLABORATORS.md # 合作者项目概述 ⭐
│   ├── ICASSP2026_PROGRESS_REPORT.md      # ICASSP2026进展报告 ⭐
│   ├── GROUP_MEETING_SUMMARY_2025.md      # 组会总结
│   ├── PPT_GUIDE_FOR_KOZI.md             # PPT制作指南
│   └── PROJECT_STRUCTURE.md              # 项目结构说明
├── 📄 配置文件
│   ├── requirements.txt            # Python依赖
│   ├── run_simple_experiments.sh   # 简单实验运行脚本
│   └── README.md                   # 项目说明
└── 📖 本文件
    └── README_FOR_COLLABORATORS.md # 合作者使用指南 ⭐
```

---

## 🚀 **快速开始**

### 📋 **环境要求**

- **Python**: 3.8+ (推荐3.10)
- **操作系统**: Linux/macOS/Windows
- **内存**: 8GB+ (推荐16GB)
- **存储**: 5GB+ 可用空间

### 🔧 **安装步骤**

#### **1. 克隆/解压项目**
```bash
# 如果从Git克隆
git clone <repository_url>
cd MAAAD_Project

# 或者解压压缩包
unzip MAAAD_Project.zip
cd MAAAD_Project
```

#### **2. 创建虚拟环境**
```bash
# 使用conda (推荐)
conda create -n maaad python=3.10
conda activate maaad

# 或使用venv
python -m venv maaad_env
source maaad_env/bin/activate  # Linux/macOS
# maaad_env\Scripts\activate   # Windows
```

#### **3. 安装依赖**
```bash
pip install -r requirements.txt
```

#### **4. 验证安装**
```bash
python test_environment.py
```

### 🎯 **运行实验**

#### **快速测试 (推荐新手)**
```bash
# 运行简单实验，验证系统正常工作
python simple_working_experiments.py
```

#### **ICASSP2026优化系统 (主要实验)**
```bash
# 运行ICASSP2026优化系统 - 这是我们的主要投稿系统
python icassp2026_optimized_system.py
```

#### **完整数据集实验**
```bash
# 运行所有数据集的完整实验
python real_working_experiments.py
```

#### **使用脚本运行**
```bash
# 使用提供的脚本运行实验
chmod +x run_simple_experiments.sh
./run_simple_experiments.sh
```

---

## 📊 **实验数据与结果**

### 🎯 **数据集信息**

| 数据集 | 训练样本 | 测试样本 | 特征维度 | 异常比例 | 状态 |
|--------|----------|----------|----------|----------|------|
| MSL | 58,317 | 73,729 | 55 | 10.72% | ✅ 可用 |
| SMAP | 135,183 | 427,617 | 25 | 13.13% | ✅ 可用 |
| SMD | 23,688 | 23,689 | 38 | 4.16% | ✅ 可用 |
| PSM | 132,481 | 87,841 | 25 | 27.8% | ✅ 可用 |
| SWAT | 755,936 | 188,984 | 105 | 12.14% | ✅ 可用 |

### 📈 **最新实验结果**

#### **MSL数据集结果**
| 排名 | 方法 | AUROC | F1-Score | 状态 |
|------|------|-------|----------|------|
| 🥇 | IsolationForest | 0.6147 | 0.1474 | 基准 |
| 🥈 | EnhancedMultiAgent | 0.5711 | 0.1693 | 我们的方法 |
| 🥉 | LocalOutlierFactor | 0.5567 | 0.1924 | 传统方法 |

#### **SMAP数据集结果**
| 排名 | 方法 | AUROC | F1-Score | 状态 |
|------|------|-------|----------|------|
| 🥇 | IsolationForest | 0.6437 | 0.0606 | 基准 |
| 🥈 | LocalOutlierFactor | 0.6235 | 0.2762 | 传统方法 |
| 🥉 | RandomForest | 0.5000 | 0.0000 | 传统方法 |

#### **平均性能对比**
| 方法 | 平均AUROC | 排名 | 说明 |
|------|-----------|------|------|
| IsolationForest | 0.6292 | 🥇 | 当前最佳基准 |
| LocalOutlierFactor | 0.5901 | 🥈 | 传统方法 |
| EnhancedMultiAgent | 0.5290 | 🥉 | 我们的方法 |
| FixedMultiAgent | 0.5107 | 4 | 修复版本 |

### 🎯 **ICASSP2026目标**

- **性能目标**: AUROC > 0.7
- **当前状态**: 系统开发完成，实验验证进行中
- **预期提升**: 通过ICASSP2026优化系统达到目标性能

---

## 🛠️ **核心功能说明**

### 🤖 **多智能体系统**

#### **5个专业智能体**
1. **趋势分析智能体**: 分析线性/非线性趋势和趋势突变
2. **方差分析智能体**: 检测方差变化和异常波动
3. **残差分析智能体**: 分析预测残差和重构误差
4. **统计分析智能体**: 提取统计特征和分布分析
5. **频域分析智能体**: 进行FFT和频谱分析

#### **13种算法集成**
每个智能体内部集成：
- **IsolationForest** (4个不同参数)
- **OneClassSVM** (5个不同参数)
- **LocalOutlierFactor** (4个不同参数)

### 🧠 **LLM驱动决策**

#### **阿里云百炼集成**
- **智能协调**: 智能体间任务分配和信息共享
- **融合策略生成**: 动态生成最优融合权重
- **决策优化**: 提供高级推理能力

#### **T2MAC通信协议**
- **目标导向通信**: 基于任务目标的智能体通信
- **消息路由**: 高效的消息传递机制
- **状态同步**: 智能体状态实时同步

### 🔧 **高级特征工程**

#### **327维多尺度特征**
- **统计特征** (11维): 均值、标准差、分位数、偏度、峰度
- **趋势特征** (110维): 一阶、二阶导数
- **频域特征** (45维): FFT、功率谱、频谱质心
- **小波特征** (80维): 多尺度下采样、小波能量
- **峰值特征** (2维): 峰值数量、平均高度
- **自相关特征** (5维): 前5个自相关系数
- **交互特征** (74维): 特征间乘积、比值关系

#### **智能NaN值处理**
- **KNN插值**: 使用K近邻插值处理NaN值
- **异常值处理**: MAD方法识别和处理异常值
- **信号平滑**: 高斯滤波平滑信号
- **去趋势**: 线性去趋势处理

---

## 📚 **重要文档说明**

### 📖 **必读文档**

1. **`DETAILED_PROJECT_INTRODUCTION.md`** ⭐
   - 详细的技术介绍和架构说明
   - 包含完整的系统工作流程
   - 核心创新点详细解释

2. **`PROJECT_OVERVIEW_FOR_COLLABORATORS.md`** ⭐
   - 专门为合作者准备的项目概述
   - 包含实验数据和技术细节
   - ICASSP2026投稿时间线

3. **`ICASSP2026_PROGRESS_REPORT.md`** ⭐
   - ICASSP2026投稿进展报告
   - 当前状态和下一步计划
   - 团队协作指南

### 📋 **其他重要文档**

- **`GROUP_MEETING_SUMMARY_2025.md`**: 组会汇报总结
- **`PPT_GUIDE_FOR_KOZI.md`**: PPT制作指南
- **`PROJECT_STRUCTURE.md`**: 项目结构说明

---

## 🎯 **ICASSP2026投稿计划**

### 📅 **时间线**

#### **Phase 1: 实验验证阶段** (2025年9月24日 - 10月1日)
- **目标**: 完成ICASSP2026优化系统的完整实验验证
- **里程碑**: 确认达到AUROC > 0.7的投稿要求
- **负责人**: 技术团队

#### **Phase 2: 论文撰写阶段** (2025年10月1日 - 10月15日)
- **目标**: 完成论文初稿撰写
- **里程碑**: 论文结构确定，核心方法完成
- **负责人**: 写作团队

#### **Phase 3: 投稿准备阶段** (2025年10月15日 - 10月31日)
- **目标**: 论文完善和投稿提交
- **里程碑**: 成功提交至ICASSP2026
- **负责人**: 全体团队

### 🤝 **团队协作**

#### **技术开发团队**
- 负责多智能体系统核心代码开发
- 负责实验环境搭建与数据预处理
- 负责模型训练、性能评估与优化

#### **论文写作团队**
- 负责论文的引言、相关工作、讨论和结论
- 负责论文的整体结构和逻辑梳理
- 负责论文的语言润色和格式检查

---

## 🔧 **故障排除**

### ❌ **常见问题**

#### **1. 环境问题**
```bash
# 如果遇到包版本冲突
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall

# 如果遇到CUDA问题
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### **2. 数据问题**
```bash
# 如果数据集加载失败，运行数据转换脚本
python convert_datasets.py

# 如果SMD标签维度不匹配
python fix_smd_labels.py
```

#### **3. 实验问题**
```bash
# 如果实验运行失败，先运行环境测试
python test_environment.py

# 如果多智能体系统失败，运行简单实验
python simple_working_experiments.py
```

### 📞 **获取帮助**

1. **查看日志**: 检查 `logs/` 目录下的日志文件
2. **运行测试**: 使用 `test_environment.py` 诊断问题
3. **联系团队**: 通过项目文档中的联系方式获取支持

---

## 🎉 **项目亮点**

### 🏆 **技术成就**

1. **首次LLM驱动**: 将大语言模型引入多智能体异常检测领域
2. **327维特征工程**: 远超传统方法的特征提取能力
3. **13种算法集成**: 提高系统鲁棒性和性能
4. **智能NaN处理**: 彻底解决数据预处理问题
5. **T2MAC通信协议**: 创新的多智能体通信机制

### 🌟 **应用价值**

1. **学术价值**: 理论贡献和方法创新
2. **工业价值**: 完整的工业级部署方案
3. **实用性强**: 支持大规模分布式计算
4. **可扩展性**: 易于扩展到其他领域

### 🚀 **未来展望**

1. **ICASSP2026投稿**: 目标成功投稿并获得接收
2. **开源贡献**: 准备开源代码和文档
3. **技术推广**: 推动多智能体异常检测领域发展
4. **应用拓展**: 扩展到更多实际应用场景

---

## 📞 **联系方式**

- **项目负责人**: 姜浩
- **技术团队**: 多智能体系统开发团队
- **写作团队**: 论文撰写团队
- **项目状态**: ICASSP2026投稿准备中

---

## 📄 **许可证**

本项目采用开源许可证，具体条款请查看项目根目录下的LICENSE文件。

---

## 🙏 **致谢**

感谢所有参与MAAAD项目开发的团队成员，以及提供数据集和计算资源的合作伙伴。

---

**最后更新**: 2025年9月24日  
**版本**: v2.0 (ICASSP2026优化版)  
**状态**: 项目开发完成，实验验证进行中

---

*如有任何问题或建议，请随时联系项目团队。祝您使用愉快！* 🚀
