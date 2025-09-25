# MAAAD项目结构说明

## 目录结构

```
maaad_project/
├── README.md                    # 项目说明
├── requirements.txt             # 依赖包列表
├── .gitignore                  # Git忽略文件
├── PROJECT_STRUCTURE.md        # 项目结构说明
│
├── src/                        # 源代码目录
│   ├── agents/                 # 智能体实现
│   │   ├── base_agent.py       # 智能体基类
│   │   ├── multi_agent_detector.py  # 多智能体检测器
│   │   └── trend_agent.py      # 趋势分析智能体
│   ├── communication/          # 通信协议
│   │   ├── t2mac_protocol.py   # T2MAC协议
│   │   └── llm_driven_communication.py  # LLM驱动通信
│   ├── llm/                    # LLM接口
│   │   └── qwen_interface.py   # Qwen LLM接口
│   ├── privacy/                # 隐私保护
│   │   └── differential_privacy.py  # 差分隐私机制
│   ├── federated/              # 联邦学习
│   │   └── federated_learning.py    # 联邦学习框架
│   └── utils/                  # 工具函数
│
├── experiments/                # 实验脚本目录
│   ├── ablation/               # 消融研究
│   │   └── quick_ablation_fix.py
│   ├── multi_dataset/          # 多数据集实验
│   │   └── multi_dataset_experiments.py
│   ├── complexity/             # 计算复杂度分析
│   │   └── computational_complexity_analysis.py
│   ├── real_world/             # 真实世界实验
│   ├── test_multi_agent.py     # 多智能体测试
│   ├── test_integration_complete.py  # 集成测试
│   ├── test_federated_learning.py    # 联邦学习测试
│   ├── test_gpu_compatibility.py     # GPU兼容性测试
│   └── test_llm_communication.py     # LLM通信测试
│
├── configs/                    # 配置文件目录
│   ├── agent_config.yaml       # 智能体配置
│   ├── server_compatibility_config.yaml  # 服务器兼容性配置
│   ├── server_experiment_config.json     # 服务器实验配置
│   └── server_optimized_config.yaml     # 服务器优化配置
│
├── docs/                       # 文档目录
│   ├── FINAL_PROJECT_SUMMARY.md        # 项目总结
│   ├── PROJECT_STATUS_REPORT.md        # 项目状态报告
│   ├── SERVER_DEPLOYMENT_GUIDE.md      # 服务器部署指南
│   ├── MANUAL_DEPLOYMENT_GUIDE.md      # 手动部署指南
│   ├── DEPLOYMENT_EXECUTION_GUIDE.md   # 部署执行指南
│   ├── SERVER_BRIEF.md                 # 服务器简介
│   └── WEEK1_SUMMARY_REPORT.md         # 第一周总结报告
│
├── scripts/                    # 脚本目录
│   ├── deploy_to_server.sh     # 服务器部署脚本
│   ├── deploy_with_ssh_key.sh  # SSH密钥部署脚本
│   ├── sync_to_server.sh       # 服务器同步脚本
│   ├── create_server_package.py        # 创建服务器包
│   └── cross_platform_setup.py         # 跨平台设置
│
├── outputs/                    # 输出结果目录
│   └── group_meeting_materials/        # 组会汇报材料
│       ├── group_meeting_summary_*.md  # 组会汇报总结
│       └── presentation_slides_*.md    # 演示幻灯片
│
├── data/                       # 数据文件目录
│
└── logs/                       # 日志文件目录
    └── project_logs/           # 项目日志
        ├── 2025-08-13.md       # 8月13日日志
        ├── 2025-08-14.md       # 8月14日日志
        ├── PLAN.md              # 计划文档
        ├── README.md            # 日志说明
        └── TEMPLATE.md          # 日志模板
```

## 文件说明

### 核心源代码 (src/)
- **agents/**: 智能体相关实现
- **communication/**: 通信协议和LLM驱动通信
- **llm/**: 大语言模型接口
- **privacy/**: 隐私保护机制
- **federated/**: 联邦学习框架

### 实验脚本 (experiments/)
- **ablation/**: 消融研究实验
- **multi_dataset/**: 多数据集验证实验
- **complexity/**: 计算复杂度分析实验
- **real_world/**: 真实世界数据实验

### 配置文件 (configs/)
- 各种YAML和JSON配置文件
- 服务器部署配置
- 实验参数配置

### 文档 (docs/)
- 项目总结和状态报告
- 部署指南和说明文档
- 技术文档和API说明

### 脚本 (scripts/)
- 部署和同步脚本
- 跨平台兼容性脚本
- 自动化工具

### 输出结果 (outputs/)
- 实验输出和结果
- 组会汇报材料
- 图表和数据文件

## 使用说明

1. **开发**: 在 `src/` 目录下进行代码开发
2. **实验**: 在 `experiments/` 目录下运行各种实验
3. **配置**: 在 `configs/` 目录下修改配置参数
4. **部署**: 使用 `scripts/` 目录下的脚本进行部署
5. **文档**: 查看 `docs/` 目录下的相关文档

## 注意事项

- 所有实验输出应保存到 `outputs/` 目录
- 配置文件不应包含敏感信息
- 日志文件应定期清理
- 数据文件应使用 `.gitignore` 忽略
