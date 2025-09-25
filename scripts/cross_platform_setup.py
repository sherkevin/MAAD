#!/usr/bin/env python3
"""
跨平台兼容性设置脚本
处理Mac和Linux之间的差异，确保项目在服务器上正常运行
"""

import os
import sys
import platform
import subprocess
import json
from pathlib import Path

def detect_platform():
    """检测当前平台"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    print(f"🔍 检测平台信息:")
    print(f"   系统: {system}")
    print(f"   架构: {machine}")
    print(f"   Python版本: {sys.version}")
    
    if system == "darwin":
        return "mac"
    elif system == "linux":
        return "linux"
    elif system == "windows":
        return "windows"
    else:
        return "unknown"

def check_gpu_availability():
    """检查GPU可用性"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"🎮 GPU状态: {'可用' if cuda_available else '不可用'}")
        
        if cuda_available:
            print(f"   CUDA版本: {torch.version.cuda}")
            print(f"   GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        
        return cuda_available
    except ImportError:
        print("❌ PyTorch未安装")
        return False

def create_platform_config():
    """创建平台特定配置"""
    platform_type = detect_platform()
    
    config = {
        "platform": platform_type,
        "gpu_available": check_gpu_availability(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "paths": {
            "project_root": str(Path.cwd()),
            "src_dir": "src",
            "config_dir": "configs",
            "output_dir": "outputs"
        }
    }
    
    # 平台特定设置
    if platform_type == "mac":
        config["device_preference"] = "cpu"  # Mac通常使用CPU
        config["torch_backend"] = "mps" if check_gpu_availability() else "cpu"
        config["batch_size"] = 32  # Mac内存限制
    elif platform_type == "linux":
        config["device_preference"] = "cuda" if check_gpu_availability() else "cpu"
        config["torch_backend"] = "cuda" if check_gpu_availability() else "cpu"
        config["batch_size"] = 64  # Linux服务器通常有更多内存
    else:
        config["device_preference"] = "cpu"
        config["torch_backend"] = "cpu"
        config["batch_size"] = 32
    
    return config

def create_requirements_file():
    """创建requirements文件"""
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "numpy<2.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.1.0",
        "matplotlib>=3.5.0",
        "pandas>=1.4.0",
        "hydra-core>=1.2.0",
        "omegaconf>=2.2.0",
        "anomalib>=0.6.0",
        "statsmodels>=0.13.0",
        "python-dotenv>=0.19.0"
    ]
    
    # 根据平台添加特定依赖
    platform_type = detect_platform()
    if platform_type == "linux":
        requirements.extend([
            "nvidia-ml-py3>=7.352.0",  # GPU监控
            "psutil>=5.9.0"  # 系统监控
        ])
    
    return requirements

def create_server_script():
    """创建服务器运行脚本"""
    script_content = '''#!/bin/bash
# 服务器运行脚本
# 自动检测环境并运行实验

echo "🚀 启动多智能体协同异常检测系统"
echo "=================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3未安装"
    exit 1
fi

# 检查CUDA环境
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA驱动已安装"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
else
    echo "⚠️  NVIDIA驱动未安装，将使用CPU"
fi

# 激活虚拟环境（如果存在）
if [ -d "venv" ]; then
    echo "🔧 激活虚拟环境"
    source venv/bin/activate
elif [ -d "conda_env" ]; then
    echo "🔧 激活Conda环境"
    source conda_env/bin/activate
fi

# 安装依赖
echo "📦 安装依赖包"
pip install -r requirements.txt

# 运行环境检查
echo "🔍 运行环境检查"
python3 check_gpu_environment.py

# 运行GPU兼容性测试
echo "🧪 运行GPU兼容性测试"
python3 test_gpu_compatibility.py

# 运行完整测试
echo "🧪 运行完整系统测试"
python3 test_integration_complete.py

# 运行实验
echo "🚀 开始运行实验"
python3 run_server_experiments.py --experiment all --gpu-test

echo "✅ 所有任务完成"
'''
    
    return script_content

def create_dockerfile():
    """创建Dockerfile用于容器化部署"""
    dockerfile_content = '''# 多智能体协同异常检测系统 Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    python3-dev \\
    build-essential \\
    git \\
    wget \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# 创建项目目录
WORKDIR /app

# 复制项目文件
COPY . .

# 安装Python依赖
RUN pip3 install --no-cache-dir -r requirements.txt

# 设置权限
RUN chmod +x run_server_experiments.py
RUN chmod +x check_gpu_environment.py

# 暴露端口（如果需要）
EXPOSE 8000

# 运行命令
CMD ["python3", "run_server_experiments.py", "--experiment", "all"]
'''
    
    return dockerfile_content

def create_docker_compose():
    """创建docker-compose.yml"""
    compose_content = '''version: '3.8'

services:
  maaad-server:
    build: .
    container_name: maaad-server
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./outputs:/app/outputs
      - ./configs:/app/configs
    ports:
      - "8000:8000"
    command: python3 run_server_experiments.py --experiment all
    restart: unless-stopped
'''
    
    return compose_content

def main():
    """主函数"""
    print("🔧 开始跨平台兼容性设置")
    print("=" * 50)
    
    # 检测平台
    platform_type = detect_platform()
    print(f"\n📊 当前平台: {platform_type}")
    
    # 创建配置
    config = create_platform_config()
    print(f"\n⚙️  平台配置:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # 保存配置
    with open('platform_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\n💾 平台配置已保存到 platform_config.json")
    
    # 创建requirements文件
    requirements = create_requirements_file()
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(requirements))
    print(f"📦 requirements.txt 已创建")
    
    # 创建服务器脚本
    server_script = create_server_script()
    with open('run_server.sh', 'w', encoding='utf-8') as f:
        f.write(server_script)
    os.chmod('run_server.sh', 0o755)
    print(f"🚀 服务器运行脚本已创建: run_server.sh")
    
    # 创建Docker文件
    dockerfile = create_dockerfile()
    with open('Dockerfile', 'w', encoding='utf-8') as f:
        f.write(dockerfile)
    print(f"🐳 Dockerfile 已创建")
    
    compose_file = create_docker_compose()
    with open('docker-compose.yml', 'w', encoding='utf-8') as f:
        f.write(compose_file)
    print(f"🐳 docker-compose.yml 已创建")
    
    # 创建项目同步脚本
    sync_script = f'''#!/bin/bash
# 项目同步脚本
# 从本地Mac同步到Linux服务器

echo "🔄 开始同步项目到服务器"
echo "========================="

# 设置服务器信息
SERVER_USER="your_username"
SERVER_HOST="your_server_ip"
SERVER_PATH="/path/to/project"

# 同步项目文件
echo "📁 同步项目文件..."
rsync -avz --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \\
    ./ {SERVER_USER}@{SERVER_HOST}:{SERVER_PATH}/

# 同步完成后在服务器上运行
echo "🚀 在服务器上运行项目..."
ssh {SERVER_USER}@{SERVER_HOST} "cd {SERVER_PATH} && chmod +x run_server.sh && ./run_server.sh"

echo "✅ 同步完成"
'''
    
    with open('sync_to_server.sh', 'w', encoding='utf-8') as f:
        f.write(sync_script)
    os.chmod('sync_to_server.sh', 0o755)
    print(f"🔄 服务器同步脚本已创建: sync_to_server.sh")
    
    print(f"\n🎉 跨平台兼容性设置完成！")
    print(f"📋 下一步:")
    print(f"   1. 编辑 sync_to_server.sh 设置服务器信息")
    print(f"   2. 运行 ./sync_to_server.sh 同步到服务器")
    print(f"   3. 在服务器上运行 ./run_server.sh")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
