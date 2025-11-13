#!/bin/bash

# WarmTalk 启动脚本

echo "🚀 启动 WarmTalk..."
echo ""

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 Python3，请先安装 Python 3.8+"
    exit 1
fi

# 检查依赖
if [ ! -d "venv" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
fi

echo "📥 激活虚拟环境并安装依赖..."
source venv/bin/activate
#pip install -r requirements.txt

echo ""
echo "✅ 依赖安装完成"
echo ""
echo "🌐 启动服务器..."
echo "访问地址: http://localhost:8000"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

python -m app.main

