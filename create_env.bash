#!/bin/bash
# 设置M2 Max优化的深度学习环境

# 显示设置信息
echo "======================================================"
echo "     设置M2 Max优化的深度学习环境"
echo "======================================================"

# 检测系统
echo "检测系统..."
if [[ $(uname -m) == 'arm64' ]]; then
    echo "✅ 检测到Apple Silicon芯片"
else
    echo "⚠️ 未检测到Apple Silicon芯片，某些优化可能无效"
fi

# 创建conda环境
echo "创建conda环境..."
conda env create -f env.yml
if [ $? -ne 0 ]; then
    echo "❌ 环境创建失败，请检查错误信息"
    exit 1
fi

# 激活环境
echo "激活环境..."
eval "$(conda shell.bash hook)"
conda activate pendulum

# 检查PyTorch MPS支持
echo "检查PyTorch MPS支持..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'MPS可用: {torch.backends.mps.is_available()}'); print(f'MPS已构建: {torch.backends.mps.is_built()}')"

# 创建项目目录结构
echo "创建项目目录结构..."
mkdir -p models figures

# 检查完成
echo "======================================================"
echo "环境设置完成！"
echo "使用以下命令激活环境："
echo "conda activate pendulum"
echo "======================================================"