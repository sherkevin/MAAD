# MAAAD - Multi-Agent Anomaly Detection System

## 项目概述

MAAAD (Multi-Agent Anomaly Detection) 是一个基于多智能体协同的异常检测系统，专为顶会论文发表而设计。

## 核心特性

- **多智能体协同**: 首次将多智能体系统应用于异常检测
- **LLM驱动通信**: 使用大语言模型增强智能体间通信
- **隐私保护**: 集成差分隐私的联邦学习框架
- **实时性能**: 支持大规模实时异常检测

## 项目结构

```
maaad_project/
├── src/                    # 源代码
│   ├── agents/            # 智能体实现
│   ├── communication/     # 通信协议
│   ├── llm/              # LLM接口
│   ├── privacy/          # 隐私保护
│   ├── federated/        # 联邦学习
│   └── utils/            # 工具函数
├── experiments/           # 实验脚本
│   ├── ablation/         # 消融研究
│   ├── multi_dataset/    # 多数据集实验
│   ├── complexity/       # 计算复杂度分析
│   └── real_world/       # 真实世界实验
├── configs/              # 配置文件
├── docs/                 # 文档
├── scripts/              # 部署脚本
├── outputs/              # 输出结果
├── data/                 # 数据文件
└── logs/                 # 日志文件
```

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+ (可选)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行实验

```bash
# 消融研究
python experiments/ablation/quick_ablation_fix.py

# 多数据集实验
python experiments/multi_dataset/multi_dataset_experiments.py

# 计算复杂度分析
python experiments/complexity/computational_complexity_analysis.py
```

## 实验结果

### 消融研究
- 趋势分析智能体: AUROC 0.5951
- 多智能体组合: AUROC 0.5652

### 多数据集验证
- MSL数据集: AUROC 0.5647
- SMAP数据集: AUROC 0.4820

### 计算复杂度
- 吞吐量: 4000+ 样本/秒
- 可扩展性: 支持大规模数据处理

## 论文贡献

1. **理论贡献**: 多智能体协同异常检测框架
2. **实验贡献**: 多数据集验证和消融研究
3. **应用贡献**: 支持大规模实时异常检测

## 许可证

MIT License

## 联系方式

项目团队
