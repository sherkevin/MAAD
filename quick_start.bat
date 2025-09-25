@echo off
echo 🚀 MAAAD项目快速开始脚本
echo ================================

echo 📋 检查Python环境...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python未安装或未添加到PATH
    pause
    exit /b 1
)

echo 📦 安装依赖包...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ 依赖安装失败
    pause
    exit /b 1
)

echo 🧪 运行环境测试...
python test_environment.py
if %errorlevel% neq 0 (
    echo ❌ 环境测试失败
    pause
    exit /b 1
)

echo 🎯 运行简单实验...
python simple_working_experiments.py

echo ✅ 快速开始完成！
echo 📖 请阅读 README_FOR_COLLABORATORS.md 了解更多信息
pause
