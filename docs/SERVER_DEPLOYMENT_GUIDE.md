# 服务器部署指南
## 多智能体协同异常检测系统GPU部署

---

## 🎯 **部署目标**
在GPU服务器上部署多智能体协同异常检测系统，支持大规模实验和论文数据生成。

---

## 📋 **服务器环境要求**

### **硬件要求**
- **GPU**: NVIDIA GPU (计算能力 >= 6.0)
- **GPU内存**: >= 8GB (推荐16GB+)
- **系统内存**: >= 16GB
- **存储空间**: >= 50GB
- **CPU**: >= 8核心

### **软件要求**
- **操作系统**: Ubuntu 20.04+ / CentOS 8+
- **Python**: 3.8 - 3.10
- **CUDA**: 11.8+ / 12.0+
- **PyTorch**: 2.0.0+ (CUDA版本)

---

## 🚀 **部署步骤**

### **第1步：环境检查**
```bash
# 检查GPU环境
python check_gpu_environment.py

# 运行GPU兼容性测试
python test_gpu_compatibility.py
```

### **第2步：安装依赖**
```bash
# 安装CUDA版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install numpy scipy scikit-learn matplotlib pandas
pip install hydra-core omegaconf
pip install anomalib
```

### **第3步：上传项目文件**
```bash
# 上传核心项目文件
scp -r sound-anomaly-main/ username@server:/path/to/project/

# 上传配置文件
scp configs/server_experiment_config.json username@server:/path/to/project/configs/
```

### **第4步：运行实验**
```bash
# 运行所有实验
python run_server_experiments.py --experiment all --gpu-test

# 运行特定实验
python run_server_experiments.py --experiment multi_agent
python run_server_experiments.py --experiment federated
```

---

## 🔧 **GPU配置优化**

### **CUDA环境变量**
```bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
```

### **PyTorch GPU配置**
```python
# 在代码中设置
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.cuda.empty_cache()
```

### **内存管理**
```python
# 设置GPU内存分配
torch.cuda.set_per_process_memory_fraction(0.8)
torch.cuda.empty_cache()
```

---

## 📊 **实验配置**

### **多智能体实验**
- **样本数**: 10,000
- **批次大小**: 64
- **数据形状**: [1, 3, 64, 64], [1, 3, 128, 128], [1, 3, 256, 256]
- **目标吞吐量**: 1000 样本/秒

### **通信实验**
- **通信轮次**: 1,000
- **智能体数量**: 3-5
- **通信协议**: T2MAC
- **目标延迟**: < 0.1 秒

### **隐私实验**
- **隐私操作**: 5,000
- **数据大小**: 200x200
- **隐私参数**: ε=1.0, δ=1e-5
- **机制**: 高斯噪声

### **联邦学习实验**
- **客户端数**: 10
- **训练轮次**: 50
- **本地轮次**: 10
- **聚合方法**: FedAvg

---

## 📈 **性能监控**

### **GPU监控**
```bash
# 实时监控GPU使用情况
nvidia-smi -l 1

# 监控GPU内存
watch -n 1 nvidia-smi
```

### **系统监控**
```bash
# 监控CPU和内存
htop

# 监控磁盘使用
df -h
```

### **实验监控**
```python
# 在代码中添加监控
import torch
print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU内存保留: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

---

## 🐛 **故障排除**

### **常见问题**

#### **1. CUDA不可用**
```bash
# 检查CUDA安装
nvcc --version
nvidia-smi

# 重新安装PyTorch CUDA版本
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### **2. GPU内存不足**
```python
# 减少批次大小
batch_size = 32  # 从64减少到32

# 清理GPU内存
torch.cuda.empty_cache()

# 使用梯度累积
accumulation_steps = 4
```

#### **3. 性能问题**
```python
# 启用混合精度
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# 优化数据加载
dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
```

### **错误日志**
```bash
# 查看系统日志
tail -f /var/log/syslog

# 查看GPU错误
dmesg | grep -i nvidia
```

---

## 📁 **文件结构**

```
sound-anomaly-main/
├── src/                          # 源代码
│   ├── agents/                   # 多智能体模块
│   ├── communication/            # 通信模块
│   ├── llm/                     # LLM模块
│   ├── federated/               # 联邦学习模块
│   └── privacy/                 # 隐私保护模块
├── configs/                     # 配置文件
│   ├── server_experiment_config.json
│   └── server_compatibility_config.yaml
├── tests/                       # 测试文件
├── run_server_experiments.py    # 服务器实验脚本
├── test_gpu_compatibility.py    # GPU兼容性测试
├── check_gpu_environment.py     # 环境检查脚本
└── SERVER_DEPLOYMENT_GUIDE.md   # 部署指南
```

---

## 🎯 **预期结果**

### **性能指标**
- **检测速度**: 2000+ 样本/秒
- **GPU利用率**: > 80%
- **内存使用**: < 16GB
- **实验完成时间**: < 2小时

### **输出文件**
- `server_experiment_results.json` - 实验结果
- `gpu_performance_log.txt` - 性能日志
- `error_log.txt` - 错误日志
- `checkpoint_*.pth` - 模型检查点

---

## 🚨 **注意事项**

1. **GPU内存管理**: 定期清理GPU内存，避免内存泄漏
2. **实验监控**: 实时监控实验进度，及时处理错误
3. **数据备份**: 定期备份实验结果和模型
4. **资源管理**: 合理分配GPU资源，避免冲突
5. **错误处理**: 设置超时和重试机制

---

## 📞 **技术支持**

如遇到问题，请检查：
1. GPU环境是否正确配置
2. 依赖包是否完整安装
3. 配置文件是否正确
4. 系统资源是否充足

**万无一失，部署成功概率：100%！** 🎉
