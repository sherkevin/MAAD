#!/bin/bash
echo "🚀 MAAAD项目快速开始脚本"
echo "================================"

echo "📋 检查Python环境..."
python3 --version || python --version
if [ $? -ne 0 ]; then
    echo "❌ Python未安装或未添加到PATH"
    exit 1
fi

echo "📦 安装依赖包..."
pip3 install -r requirements.txt || pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ 依赖安装失败"
    exit 1
fi

echo "🧪 运行环境测试..."
python3 test_environment.py || python test_environment.py
if [ $? -ne 0 ]; then
    echo "❌ 环境测试失败"
    exit 1
fi

echo "🎯 运行简单实验..."
python3 simple_working_experiments.py || python simple_working_experiments.py

echo "✅ 快速开始完成！"
echo "📖 请阅读 README_FOR_COLLABORATORS.md 了解更多信息"
