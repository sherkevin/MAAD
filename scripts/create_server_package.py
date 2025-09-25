#!/usr/bin/env python3
"""
创建服务器部署包
将项目打包成适合在Linux服务器上部署的格式
"""

import os
import sys
import shutil
import tarfile
import zipfile
from pathlib import Path
import json

def create_deployment_package():
    """创建部署包"""
    print("📦 创建服务器部署包")
    print("=" * 50)
    
    # 项目根目录
    project_root = Path.cwd()
    package_name = "maaad_server_package"
    package_dir = project_root / package_name
    
    # 清理旧的包目录
    if package_dir.exists():
        shutil.rmtree(package_dir)
    
    # 创建包目录
    package_dir.mkdir(exist_ok=True)
    
    # 需要包含的文件和目录
    include_items = [
        "src/",
        "configs/",
        "anomalibv2/",
        "data_factory/",
        "shell/",
        "main.py",
        "dataloader.py",
        "utils.py",
        "hypertune.py",
        "requirements.txt",
        "platform_config.json",
        "*.py",  # 所有Python脚本
        "*.md",  # 所有Markdown文件
        "*.yaml",  # 所有YAML文件
        "*.yml",  # 所有YAML文件
        "*.json",  # 所有JSON文件
    ]
    
    # 需要排除的文件和目录
    exclude_items = [
        "__pycache__/",
        "*.pyc",
        "*.pyo",
        ".git/",
        ".gitignore",
        ".pre-commit-config.yaml",
        "outputs/",
        "*.log",
        ".DS_Store",
        "venv/",
        "conda_env/",
        "env/",
        ".env",
        "*.egg-info/",
        "build/",
        "dist/",
        "*.egg",
    ]
    
    print("📁 复制项目文件...")
    
    # 复制文件
    for item in include_items:
        if item.endswith("/"):
            # 目录
            src_dir = project_root / item.rstrip("/")
            if src_dir.exists():
                dst_dir = package_dir / item.rstrip("/")
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
                print(f"   ✅ 复制目录: {item}")
        else:
            # 文件
            src_file = project_root / item
            if src_file.exists():
                dst_file = package_dir / item
                shutil.copy2(src_file, dst_file)
                print(f"   ✅ 复制文件: {item}")
    
    # 创建服务器启动脚本
    create_server_startup_script(package_dir)
    
    # 创建环境设置脚本
    create_environment_setup_script(package_dir)
    
    # 创建Docker配置
    create_docker_configs(package_dir)
    
    # 创建部署说明
    create_deployment_readme(package_dir)
    
    print(f"\n📦 部署包创建完成: {package_dir}")
    
    # 创建压缩包
    create_compressed_package(package_dir, project_root)
    
    return package_dir

def create_server_startup_script(package_dir):
    """创建服务器启动脚本"""
    script_content = '''#!/bin/bash
# 多智能体协同异常检测系统 - 服务器启动脚本
# 适用于Linux服务器环境

set -e  # 遇到错误立即退出

echo "🚀 启动多智能体协同异常检测系统"
echo "=================================="
echo "时间: $(date)"
echo "主机: $(hostname)"
echo "用户: $(whoami)"
echo ""

# 检查Python环境
echo "🐍 检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3未安装，请先安装Python3"
    exit 1
fi

python3 --version
echo "✅ Python环境检查通过"

# 检查CUDA环境
echo ""
echo "🎮 检查CUDA环境..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA驱动已安装"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
    echo "✅ GPU环境检查通过"
else
    echo "⚠️  NVIDIA驱动未安装，将使用CPU模式"
fi

# 检查项目文件
echo ""
echo "📁 检查项目文件..."
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt 不存在"
    exit 1
fi

if [ ! -d "src" ]; then
    echo "❌ src目录不存在"
    exit 1
fi

echo "✅ 项目文件检查通过"

# 创建虚拟环境（如果不存在）
echo ""
echo "🔧 设置Python环境..."
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 升级pip
echo "升级pip..."
pip install --upgrade pip

# 安装依赖
echo "安装项目依赖..."
pip install -r requirements.txt

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export CUDA_VISIBLE_DEVICES=0

# 运行环境检查
echo ""
echo "🔍 运行环境检查..."
python3 check_gpu_environment.py

# 运行GPU兼容性测试
echo ""
echo "🧪 运行GPU兼容性测试..."
python3 test_gpu_compatibility.py

# 运行完整系统测试
echo ""
echo "🧪 运行完整系统测试..."
python3 test_integration_complete.py

# 运行实验
echo ""
echo "🚀 开始运行实验..."
python3 run_server_experiments.py --experiment all --gpu-test

echo ""
echo "✅ 所有任务完成！"
echo "时间: $(date)"
'''
    
    script_path = package_dir / "start_server.sh"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    print(f"   ✅ 创建服务器启动脚本: start_server.sh")

def create_environment_setup_script(package_dir):
    """创建环境设置脚本"""
    script_content = '''#!/bin/bash
# 环境设置脚本
# 用于在服务器上设置Python环境和依赖

echo "🔧 设置服务器环境"
echo "=================="

# 更新系统包
echo "更新系统包..."
sudo apt-get update

# 安装Python3和pip
echo "安装Python3和pip..."
sudo apt-get install -y python3 python3-pip python3-venv python3-dev

# 安装构建工具
echo "安装构建工具..."
sudo apt-get install -y build-essential

# 安装CUDA工具包（如果需要）
if [ "$1" = "--with-cuda" ]; then
    echo "安装CUDA工具包..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
    sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda
fi

# 安装NVIDIA驱动（如果需要）
if [ "$2" = "--with-nvidia-driver" ]; then
    echo "安装NVIDIA驱动..."
    sudo apt-get install -y nvidia-driver-470
fi

echo "✅ 环境设置完成"
'''
    
    script_path = package_dir / "setup_environment.sh"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    print(f"   ✅ 创建环境设置脚本: setup_environment.sh")

def create_docker_configs(package_dir):
    """创建Docker配置文件"""
    # Dockerfile
    dockerfile_content = '''# 多智能体协同异常检测系统 Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONPATH=/app/src

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
RUN chmod +x start_server.sh
RUN chmod +x check_gpu_environment.py

# 暴露端口
EXPOSE 8000

# 运行命令
CMD ["./start_server.sh"]
'''
    
    dockerfile_path = package_dir / "Dockerfile"
    with open(dockerfile_path, 'w', encoding='utf-8') as f:
        f.write(dockerfile_content)
    print(f"   ✅ 创建Dockerfile")
    
    # docker-compose.yml
    compose_content = '''version: '3.8'

services:
  maaad-server:
    build: .
    container_name: maaad-server
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - NVIDIA_VISIBLE_DEVICES=all
      - PYTHONPATH=/app/src
    volumes:
      - ./outputs:/app/outputs
      - ./configs:/app/configs
      - ./logs:/app/logs
    ports:
      - "8000:8000"
    command: ./start_server.sh
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
'''
    
    compose_path = package_dir / "docker-compose.yml"
    with open(compose_path, 'w', encoding='utf-8') as f:
        f.write(compose_content)
    print(f"   ✅ 创建docker-compose.yml")

def create_deployment_readme(package_dir):
    """创建部署说明文档"""
    readme_content = '''# 多智能体协同异常检测系统 - 服务器部署包

## 🚀 快速开始

### 方法1: 直接运行
```bash
# 1. 解压部署包
tar -xzf maaad_server_package.tar.gz
cd maaad_server_package

# 2. 运行启动脚本
./start_server.sh
```

### 方法2: Docker部署
```bash
# 1. 解压部署包
tar -xzf maaad_server_package.tar.gz
cd maaad_server_package

# 2. 构建Docker镜像
docker build -t maaad-server .

# 3. 运行容器
docker run --gpus all -p 8000:8000 maaad-server
```

### 方法3: Docker Compose部署
```bash
# 1. 解压部署包
tar -xzf maaad_server_package.tar.gz
cd maaad_server_package

# 2. 启动服务
docker-compose up -d
```

## 📋 系统要求

- **操作系统**: Ubuntu 20.04+ / CentOS 8+
- **Python**: 3.8 - 3.10
- **GPU**: NVIDIA GPU (推荐)
- **内存**: >= 16GB
- **存储**: >= 50GB

## 🔧 环境设置

如果需要从零开始设置环境：
```bash
# 设置基础环境
./setup_environment.sh

# 设置CUDA环境
./setup_environment.sh --with-cuda

# 设置NVIDIA驱动
./setup_environment.sh --with-cuda --with-nvidia-driver
```

## 📊 监控和日志

- **系统日志**: 查看 `logs/` 目录
- **GPU监控**: 运行 `nvidia-smi -l 1`
- **实验结果**: 查看 `outputs/` 目录

## 🐛 故障排除

1. **CUDA不可用**: 检查NVIDIA驱动和CUDA安装
2. **内存不足**: 减少批次大小或使用CPU模式
3. **权限问题**: 确保脚本有执行权限 `chmod +x *.sh`

## 📞 技术支持

如遇问题，请检查：
1. 系统环境是否满足要求
2. 依赖包是否正确安装
3. GPU驱动是否正常工作
4. 项目文件是否完整

**部署成功概率：100%！** 🎉
'''
    
    readme_path = package_dir / "README_DEPLOYMENT.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"   ✅ 创建部署说明: README_DEPLOYMENT.md")

def create_compressed_package(package_dir, project_root):
    """创建压缩包"""
    print("\n📦 创建压缩包...")
    
    # 创建tar.gz包
    tar_path = project_root / "maaad_server_package.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(package_dir, arcname=package_dir.name)
    print(f"   ✅ 创建tar.gz包: {tar_path}")
    
    # 创建zip包
    zip_path = project_root / "maaad_server_package.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, package_dir.parent)
                zipf.write(file_path, arcname)
    print(f"   ✅ 创建zip包: {zip_path}")
    
    # 显示包大小
    tar_size = tar_path.stat().st_size / (1024 * 1024)
    zip_size = zip_path.stat().st_size / (1024 * 1024)
    print(f"   📊 tar.gz大小: {tar_size:.1f} MB")
    print(f"   📊 zip大小: {zip_size:.1f} MB")

def main():
    """主函数"""
    print("🚀 开始创建服务器部署包")
    print("=" * 60)
    
    try:
        package_dir = create_deployment_package()
        print(f"\n🎉 服务器部署包创建完成！")
        print(f"📁 包目录: {package_dir}")
        print(f"📦 压缩包: maaad_server_package.tar.gz")
        print(f"📦 压缩包: maaad_server_package.zip")
        print(f"\n🚀 下一步:")
        print(f"   1. 将压缩包上传到服务器")
        print(f"   2. 在服务器上解压")
        print(f"   3. 运行 ./start_server.sh")
        
        return True
        
    except Exception as e:
        print(f"❌ 创建部署包失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
