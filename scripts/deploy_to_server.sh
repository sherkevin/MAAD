#!/bin/bash
# 一键部署脚本
# 自动上传、配置、运行服务器实验

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 检查参数
if [ $# -lt 3 ]; then
    echo "用法: $0 <用户名> <服务器IP> <服务器路径> [选项]"
    echo ""
    echo "参数:"
    echo "  用户名      服务器用户名"
    echo "  服务器IP    服务器IP地址"
    echo "  服务器路径  项目部署路径"
    echo ""
    echo "选项:"
    echo "  --with-gpu     启用GPU支持"
    echo "  --with-docker  使用Docker部署"
    echo "  --monitor      启动监控"
    echo "  --experiment   运行实验类型 (all|multi_agent|communication|privacy|federated)"
    echo ""
    echo "示例:"
    echo "  $0 ubuntu 192.168.1.100 /home/ubuntu/maaad_project"
    echo "  $0 ubuntu 192.168.1.100 /home/ubuntu/maaad_project --with-gpu --monitor"
    exit 1
fi

# 解析参数
SERVER_USER=$1
SERVER_HOST=$2
SERVER_PATH=$3
shift 3

WITH_GPU=false
WITH_DOCKER=false
WITH_MONITOR=false
EXPERIMENT_TYPE="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --with-gpu)
            WITH_GPU=true
            shift
            ;;
        --with-docker)
            WITH_DOCKER=true
            shift
            ;;
        --monitor)
            WITH_MONITOR=true
            shift
            ;;
        --experiment)
            EXPERIMENT_TYPE="$2"
            shift 2
            ;;
        *)
            print_error "未知选项: $1"
            exit 1
            ;;
    esac
done

print_info "开始一键部署到服务器"
echo "=================================="
print_info "配置信息:"
echo "  用户: $SERVER_USER"
echo "  主机: $SERVER_HOST"
echo "  路径: $SERVER_PATH"
echo "  GPU支持: $WITH_GPU"
echo "  Docker部署: $WITH_DOCKER"
echo "  监控: $WITH_MONITOR"
echo "  实验类型: $EXPERIMENT_TYPE"
echo ""

# 检查本地环境
print_info "检查本地环境..."

# 检查必要文件
required_files=(
    "src/"
    "configs/"
    "requirements.txt"
    "main.py"
    "test_integration_complete.py"
)

for file in "${required_files[@]}"; do
    if [ ! -e "$file" ]; then
        print_error "缺少必要文件: $file"
        exit 1
    fi
done
print_success "本地文件检查通过"

# 检查SSH连接
print_info "检查SSH连接..."
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes $SERVER_USER@$SERVER_HOST exit 2>/dev/null; then
    print_error "无法连接到服务器，请检查:"
    echo "  1. 服务器IP地址是否正确"
    echo "  2. 用户名是否正确"
    echo "  3. SSH密钥是否配置"
    echo "  4. 网络连接是否正常"
    exit 1
fi
print_success "SSH连接正常"

# 在服务器上创建目录
print_info "在服务器上创建目录..."
ssh $SERVER_USER@$SERVER_HOST "mkdir -p $SERVER_PATH"
print_success "目录创建完成"

# 同步项目文件
print_info "同步项目文件..."
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
    --exclude='logs/' \
    ./ $SERVER_USER@$SERVER_HOST:$SERVER_PATH/
print_success "文件同步完成"

# 在服务器上设置权限
print_info "设置文件权限..."
ssh $SERVER_USER@$SERVER_HOST "cd $SERVER_PATH && chmod +x *.sh *.py 2>/dev/null || true"
print_success "权限设置完成"

# 在服务器上安装依赖
print_info "在服务器上安装依赖..."
ssh $SERVER_USER@$SERVER_HOST "cd $SERVER_PATH && python3 -m pip install --upgrade pip && pip3 install -r requirements.txt"
print_success "依赖安装完成"

# 运行环境检查
print_info "运行环境检查..."
ssh $SERVER_USER@$SERVER_HOST "cd $SERVER_PATH && python3 check_gpu_environment.py"
print_success "环境检查完成"

# 运行基础测试
print_info "运行基础测试..."
ssh $SERVER_USER@$SERVER_HOST "cd $SERVER_PATH && python3 test_integration_complete.py"
print_success "基础测试完成"

# 根据选项进行部署
if [ "$WITH_DOCKER" = true ]; then
    print_info "使用Docker部署..."
    ssh $SERVER_USER@$SERVER_HOST "cd $SERVER_PATH && docker build -t maaad-server ."
    if [ "$WITH_GPU" = true ]; then
        ssh $SERVER_USER@$SERVER_HOST "cd $SERVER_PATH && docker run --gpus all -d --name maaad-server-container maaad-server"
    else
        ssh $SERVER_USER@$SERVER_HOST "cd $SERVER_PATH && docker run -d --name maaad-server-container maaad-server"
    fi
    print_success "Docker部署完成"
else
    print_info "直接部署..."
    # 创建虚拟环境
    ssh $SERVER_USER@$SERVER_HOST "cd $SERVER_PATH && python3 -m venv venv"
    
    # 激活虚拟环境并安装依赖
    ssh $SERVER_USER@$SERVER_HOST "cd $SERVER_PATH && source venv/bin/activate && pip install -r requirements.txt"
    
    print_success "直接部署完成"
fi

# 运行实验
print_info "开始运行实验..."
if [ "$WITH_DOCKER" = true ]; then
    if [ "$EXPERIMENT_TYPE" = "all" ]; then
        ssh $SERVER_USER@$SERVER_HOST "cd $SERVER_PATH && docker exec maaad-server-container python3 run_advanced_server_experiments.py"
    else
        ssh $SERVER_USER@$SERVER_HOST "cd $SERVER_PATH && docker exec maaad-server-container python3 run_advanced_server_experiments.py --experiment $EXPERIMENT_TYPE"
    fi
else
    if [ "$EXPERIMENT_TYPE" = "all" ]; then
        ssh $SERVER_USER@$SERVER_HOST "cd $SERVER_PATH && source venv/bin/activate && python3 run_advanced_server_experiments.py"
    else
        ssh $SERVER_USER@$SERVER_HOST "cd $SERVER_PATH && source venv/bin/activate && python3 run_advanced_server_experiments.py --experiment $EXPERIMENT_TYPE"
    fi
fi
print_success "实验运行完成"

# 启动监控（如果启用）
if [ "$WITH_MONITOR" = true ]; then
    print_info "启动监控..."
    if [ "$WITH_DOCKER" = true ]; then
        ssh $SERVER_USER@$SERVER_HOST "cd $SERVER_PATH && docker exec -d maaad-server-container python3 monitor_server_experiments.py"
    else
        ssh $SERVER_USER@$SERVER_HOST "cd $SERVER_PATH && nohup python3 monitor_server_experiments.py > monitor.log 2>&1 &"
    fi
    print_success "监控已启动"
fi

# 显示部署结果
print_success "部署完成！"
echo ""
print_info "服务器信息:"
echo "  主机: $SERVER_HOST"
echo "  路径: $SERVER_PATH"
echo "  用户: $SERVER_USER"
echo ""
print_info "访问方式:"
echo "  SSH登录: ssh $SERVER_USER@$SERVER_HOST"
echo "  项目目录: cd $SERVER_PATH"
if [ "$WITH_DOCKER" = true ]; then
    echo "  查看容器: docker ps"
    echo "  查看日志: docker logs maaad-server-container"
else
    echo "  激活环境: source venv/bin/activate"
    echo "  查看日志: tail -f monitor.log"
fi
echo ""
print_info "实验文件:"
echo "  结果文件: server_experiment_results.json"
echo "  监控日志: monitor_log.json"
echo "  系统日志: logs/"
echo ""
print_success "🎉 部署成功！万无一失！"
