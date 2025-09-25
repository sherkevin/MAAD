#!/bin/bash
# 项目同步脚本
# 从本地Mac同步到Linux服务器

echo "🔄 开始同步项目到服务器"
echo "========================="

# 设置服务器信息（请根据实际情况修改）
SERVER_USER="your_username"
SERVER_HOST="your_server_ip"
SERVER_PATH="/path/to/project"

# 检查参数
if [ $# -eq 3 ]; then
    SERVER_USER=$1
    SERVER_HOST=$2
    SERVER_PATH=$3
elif [ $# -ne 0 ]; then
    echo "用法: $0 [用户名] [服务器IP] [服务器路径]"
    echo "示例: $0 ubuntu 192.168.1.100 /home/ubuntu/maaad_project"
    exit 1
fi

echo "📊 同步配置:"
echo "   用户: $SERVER_USER"
echo "   主机: $SERVER_HOST"
echo "   路径: $SERVER_PATH"
echo ""

# 检查rsync是否可用
if ! command -v rsync &> /dev/null; then
    echo "❌ rsync未安装，请先安装rsync"
    exit 1
fi

# 检查SSH连接
echo "🔍 检查SSH连接..."
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes $SERVER_USER@$SERVER_HOST exit 2>/dev/null; then
    echo "❌ 无法连接到服务器，请检查:"
    echo "   1. 服务器IP地址是否正确"
    echo "   2. 用户名是否正确"
    echo "   3. SSH密钥是否配置"
    echo "   4. 网络连接是否正常"
    exit 1
fi
echo "✅ SSH连接正常"

# 在服务器上创建目录
echo "📁 在服务器上创建目录..."
ssh $SERVER_USER@$SERVER_HOST "mkdir -p $SERVER_PATH"

# 同步项目文件
echo "📁 同步项目文件..."
rsync -avz --progress \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='.DS_Store' \
    --exclude='venv/' \
    --exclude='conda_env/' \
    --exclude='env/' \
    --exclude='outputs/' \
    --exclude='*.log' \
    ./ $SERVER_USER@$SERVER_HOST:$SERVER_PATH/

echo "✅ 文件同步完成"

# 在服务器上设置权限
echo "🔧 设置文件权限..."
ssh $SERVER_USER@$SERVER_HOST "cd $SERVER_PATH && chmod +x *.sh && chmod +x src/*.py"

# 在服务器上运行环境检查
echo "🔍 在服务器上运行环境检查..."
ssh $SERVER_USER@$SERVER_HOST "cd $SERVER_PATH && python3 check_gpu_environment.py"

# 在服务器上运行测试
echo "🧪 在服务器上运行测试..."
ssh $SERVER_USER@$SERVER_HOST "cd $SERVER_PATH && python3 test_integration_complete.py"

echo ""
echo "✅ 同步完成！"
echo "📊 服务器信息:"
echo "   主机: $SERVER_HOST"
echo "   路径: $SERVER_PATH"
echo "   用户: $SERVER_USER"
echo ""
echo "🚀 下一步:"
echo "   1. SSH登录服务器: ssh $SERVER_USER@$SERVER_HOST"
echo "   2. 进入项目目录: cd $SERVER_PATH"
echo "   3. 运行启动脚本: ./start_server.sh"
