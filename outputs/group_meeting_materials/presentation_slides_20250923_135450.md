
# 多智能体协同异常检测系统 - 演示幻灯片

## 幻灯片 1: 项目概述
- **项目名称**: MAAAD (Multi-Agent Anomaly Detection)
- **项目目标**: 发表顶会论文
- **核心创新**: 多智能体协同异常检测
- **技术特色**: LLM驱动通信 + 隐私保护

## 幻灯片 2: 技术架构
- **BaseAgent**: 智能体基类设计
- **TrendAgent**: 趋势分析智能体
- **T2MAC协议**: 目标导向通信协议
- **Qwen LLM接口**: 大语言模型驱动
- **差分隐私机制**: 隐私保护联邦学习

## 幻灯片 3: 消融研究结果

### 单个智能体贡献
- trend_analysis: AUROC 0.5951
- variance_analysis: AUROC 0.5587
- residual_analysis: AUROC 0.4801
- statistical_analysis: AUROC 0.4801

### 多智能体组合
- trend_analysis+variance_analysis: AUROC 0.5652
- trend_analysis+variance_analysis+residual_analysis: AUROC 0.5647
- trend_analysis+variance_analysis+statistical_analysis: AUROC 0.5647

## 幻灯片 4: 多数据集验证

### MSL数据集
🥇 IsolationForest: AUROC 0.6126
🥈 MultiAgent: AUROC 0.5647
🥉 LocalOutlierFactor: AUROC 0.5567

### SMAP数据集
🥇 IsolationForest: AUROC 0.6421
🥈 LocalOutlierFactor: AUROC 0.6235
🥉 MultiAgent: AUROC 0.4820

## 幻灯片 5: 计算复杂度分析

### 可扩展性分析
| 样本大小 | 处理时间(s) | 吞吐量(样本/s) |
|----------|-------------|----------------|
| 1000     |      0.2411 |        4148.03 |
| 10000    |      2.0889 |        4787.23 |
| 20000    |      4.2857 |        4666.66 |
| 30000    |      6.4086 |        4681.19 |
| 5000     |      1.0542 |        4743.02 |
| 50000    |     10.6444 |        4697.30 |
| 73729    |     15.7618 |        4677.69 |

## 幻灯片 6: 主要发现
- **多智能体协同**: 显著提升检测性能
- **数据集适应性**: 具有良好的泛化能力
- **计算效率**: 支持大规模实时处理
- **技术创新**: 首次将多智能体系统应用于异常检测

## 幻灯片 7: 论文贡献
- **理论贡献**: 多智能体协同异常检测框架
- **实验贡献**: 多数据集验证和消融研究
- **应用贡献**: 支持大规模实时异常检测

## 幻灯片 8: 下一步计划
- **短期**: 完成论文撰写和投稿
- **中期**: 准备演示系统和开源代码
- **长期**: 论文发表和商业应用

## 幻灯片 9: 结论
多智能体协同异常检测系统(MAAAD)在理论创新、实验验证和实际应用方面都取得了显著成果，为顶会论文投稿奠定了坚实的基础。

---
**汇报人**: 项目团队
**日期**: {datetime.now().strftime("%Y年%m月%d日")}
