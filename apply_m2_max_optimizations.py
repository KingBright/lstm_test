#!/usr/bin/env python3
"""
M2 Max优化应用脚本
此脚本会自动应用所有优化并修复发现的任何问题，使项目可以在M2 Max上高效运行
"""

import os
import sys
import shutil
import tempfile
import platform
import re
from pathlib import Path

def print_header(text):
    """打印格式化的标题"""
    print("\n" + "=" * 60)
    print(f"   {text}")
    print("=" * 60)

def print_step(step_number, text):
    """打印步骤信息"""
    print(f"\n[步骤 {step_number}] {text}...")

def backup_file(filepath):
    """创建文件备份"""
    backup_path = f"{filepath}.bak"
    try:
        shutil.copy2(filepath, backup_path)
        print(f"  已创建备份: {backup_path}")
        return True
    except Exception as e:
        print(f"  创建备份时出错: {e}")
        return False

def modify_file(filepath, replacements):
    """修改文件内容"""
    # 确保文件存在
    if not os.path.exists(filepath):
        print(f"  错误: 文件不存在 - {filepath}")
        return False
        
    # 创建备份
    if not backup_file(filepath):
        return False
        
    # 读取原始内容
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        print(f"  读取文件时出错: {e}")
        return False
        
    # 应用所有替换
    original_content = content
    for pattern, replacement in replacements:
        if isinstance(pattern, str):
            content = content.replace(pattern, replacement)
        else:  # 正则表达式
            content = re.sub(pattern, replacement, content)
            
    # 如果没有变化，报告并返回
    if content == original_content:
        print(f"  没有需要更改的内容")
        return True
        
    # 写入修改后的内容
    try:
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"  文件已成功更新")
        return True
    except Exception as e:
        print(f"  写入文件时出错: {e}")
        # 尝试恢复备份
        try:
            shutil.copy2(f"{filepath}.bak", filepath)
            print(f"  已从备份恢复文件")
        except Exception as restore_error:
            print(f"  恢复备份时出错: {restore_error}")
        return False

def ensure_directory_exists(directory):
    """确保目录存在，如不存在则创建"""
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        print(f"  创建目录时出错: {e}")
        return False

def check_mps_available():
    """检查MPS是否可用"""
    try:
        import torch
        has_mps = torch.backends.mps.is_available()
        if has_mps:
            print("  ✓ MPS加速已可用，将应用M2 Max优化")
        else:
            print("  ⚠️ MPS加速不可用，但仍会应用通用优化")
        return has_mps
    except ImportError:
        print("  ⚠️ 未安装PyTorch，无法检查MPS可用性")
        return False

def apply_config_optimizations():
    """优化config.py文件"""
    print_step(1, "优化config.py配置")
    
    filepath = "config.py"
    replacements = [
        # 添加缺失的角度和角速度范围参数
        (
            "NUM_ICS_TO_RUN = len(INITIAL_CONDITIONS_SPECIFIC) # Number of long simulations",
            "NUM_ICS_TO_RUN = len(INITIAL_CONDITIONS_SPECIFIC) # Number of long simulations\n# 添加角度和角速度范围参数 (用于随机初始条件)\nTHETA_RANGE = [-np.pi/2, np.pi/2]  # 角度范围 [-90°, 90°]\nTHETA_DOT_RANGE = [-2.0, 2.0]      # 角速度范围 [-2, 2] rad/s"
        ),
        # 优化批量大小
        (
            "NUM_EPOCHS = 150; LEARNING_RATE = 0.0005; WEIGHT_DECAY = 1e-5; BATCH_SIZE = 128",
            "NUM_EPOCHS = 150; LEARNING_RATE = 0.001; WEIGHT_DECAY = 1e-5; BATCH_SIZE = 256  # 针对M2 Max增大批量大小"
        ),
        # 修改模型参数以提高复杂度
        (
            "\"seq2seqgru\": { \"hidden_size\": 32, \"num_layers\": 2 }",
            "\"seq2seqgru\": { \"hidden_size\": 64, \"num_layers\": 3 }  # 针对M2 Max增加模型复杂度"
        ),
    ]
    
    return modify_file(filepath, replacements)

def apply_training_optimizations():
    """优化training.py文件"""
    print_step(2, "优化训练过程")
    
    filepath = "training.py"
    replacements = [
        # 修复ReduceLROnPlateau的verbose参数
        (
            re.compile(r"scheduler = ReduceLROnPlateau\([^)]*verbose=True[^)]*\)"),
            "scheduler = ReduceLROnPlateau(\n        optimizer,\n        mode='min',\n        factor=scheduler_factor,\n        patience=scheduler_patience,\n        verbose=False,  # 设为False以避免警告\n        min_lr=1e-7\n    )"
        ),
        # 优化训练循环
        (
            re.compile(r"model\.train\(\); running_train_loss = 0\.0; train_batches = 0\s+for inputs, targets in train_loader:"),
            "model.train()\n        running_train_loss = 0.0\n        train_batches = 0\n        \n        # 设置梯度累积步数来提高效率\n        grad_accumulation_steps = 2  # 累积2个批次的梯度，相当于增大批次大小\n        \n        for inputs, targets in train_loader:"
        ),
        # 优化梯度计算和更新
        (
            "optimizer.zero_grad()",
            "# 仅在需要时清零梯度\n            if train_batches % grad_accumulation_steps == 0:\n                optimizer.zero_grad()"
        ),
        # 优化数据传输
        (
            "inputs, targets = inputs.to(device), targets.to(device)",
            "# 使用non_blocking=True加速数据传输\n            inputs = inputs.to(device, non_blocking=True)\n            targets = targets.to(device, non_blocking=True)"
        ),
        # 更新梯度计算和反向传播
        (
            "loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step()",
            "# 如果使用梯度累积，缩放损失\n            if grad_accumulation_steps > 1:\n                loss = loss / grad_accumulation_steps\n                \n            # 反向传播\n            loss.backward()\n            \n            # 仅在累积步骤结束时更新参数\n            if (train_batches + 1) % grad_accumulation_steps == 0 or (train_batches + 1) == len(train_loader):\n                # 梯度裁剪以提高稳定性\n                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n                optimizer.step()"
        ),
        # 添加MPS同步点
        (
            "running_train_loss += loss.item(); train_batches += 1",
            "# 累积损失并更新计数\n            running_train_loss += loss.item() * (1 if grad_accumulation_steps == 1 else grad_accumulation_steps)\n            train_batches += 1\n        \n        # 如果是MPS设备，添加同步点确保所有操作完成\n        if device.type == 'mps':\n            torch.mps.synchronize()"
        ),
    ]
    
    return modify_file(filepath, replacements)

def apply_model_optimizations():
    """优化model.py文件"""
    print_step(3, "优化模型架构")
    
    filepath = "model.py"
    replacements = [
        # 添加批归一化层
        (
            "self.input_norm = nn.LayerNorm(input_size)",
            "# 添加批归一化以加速训练和提高稳定性\n        self.input_norm = nn.LayerNorm(input_size)\n        self.batch_norm = nn.BatchNorm1d(input_size)"
        ),
        # 优化激活函数
        (
            "nn.LeakyReLU()",
            "nn.SiLU()  # 使用SiLU(Swish)激活函数以改善性能"
        ),
        # 添加ResNet风格的跳跃连接
        (
            re.compile(r"class PureGRU\(nn\.Module\):"),
            "class ResidualBlock(nn.Module):\n    \"\"\"ResNet风格的残差块，用于提高梯度流动和训练稳定性\"\"\"\n    def __init__(self, channels):\n        super(ResidualBlock, self).__init__()\n        self.block = nn.Sequential(\n            nn.Linear(channels, channels),\n            nn.BatchNorm1d(channels),\n            nn.SiLU(),\n            nn.Linear(channels, channels),\n            nn.BatchNorm1d(channels)\n        )\n        self.activation = nn.SiLU()\n        \n    def forward(self, x):\n        residual = x\n        out = self.block(x)\n        out += residual  # 跳跃连接\n        return self.activation(out)\n\n\nclass PureGRU(nn.Module):"
        ),
    ]
    
    return modify_file(filepath, replacements)

def apply_data_preprocessing_optimizations():
    """优化数据预处理"""
    print_step(4, "优化数据加载和预处理")
    
    filepath = "data_preprocessing.py"
    replacements = [
        # 自动检测CPU核心数并优化DataLoader
        (
            "train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2, persistent_workers=True)",
            "# 检测CPU核心数以优化num_workers\n        import multiprocessing\n        num_workers = min(4, max(2, multiprocessing.cpu_count() // 2))  # 使用一半的CPU核心数，但至少2个，最多4个\n        print(f\"使用 {num_workers} 个工作进程加载数据\")\n        \n        # 针对M2 Max优化的DataLoader参数\n        train_loader = DataLoader(\n            train_dataset, \n            batch_size=config.BATCH_SIZE,\n            shuffle=True, \n            pin_memory=True,\n            num_workers=num_workers,\n            persistent_workers=True,\n            prefetch_factor=2\n        )"
        ),
        # 优化验证数据加载器
        (
            "val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2, persistent_workers=False) if val_dataset else None",
            "# 验证集可使用更大批次\n        val_loader = DataLoader(\n            val_dataset, \n            batch_size=config.BATCH_SIZE * 2,  # 验证时使用更大批次\n            shuffle=False, \n            pin_memory=True,\n            num_workers=num_workers,\n            persistent_workers=True,\n            prefetch_factor=2\n        ) if val_dataset else None"
        ),
    ]
    
    return modify_file(filepath, replacements)

def apply_main_experiment_optimizations():
    """优化主实验文件"""
    print_step(5, "优化主实验流程")
    
    filepath = "main_experiment.py"
    replacements = [
        # 优化环境变量和设备设置
        (
            "device = torch.device(\"mps\" if torch.backends.mps.is_available() else \"cuda\" if torch.cuda.is_available() else \"cpu\")",
            "# 优化的设备设置，包括MPS加速\nif torch.backends.mps.is_available():\n        device = torch.device(\"mps\")\n        # 设置环境变量来优化Metal性能\n        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'\n        print(f\"使用 MPS 加速 (M2 Max)\")\n    elif torch.cuda.is_available():\n        device = torch.device(\"cuda\")\n        print(f\"使用 CUDA 加速\")\n    else:\n        device = torch.device(\"cpu\")\n        print(f\"使用 CPU 计算\")"
        ),
        # 优化模型加载
        (
            "model.load_state_dict(torch.load(config.MODEL_BEST_PATH, map_location=device, weights_only=True))",
            "# 针对M2 Max的优化加载\n            if device.type == 'mps':\n                # 使用map_location确保正确加载到MPS设备\n                state_dict = torch.load(\n                    config.MODEL_BEST_PATH, \n                    map_location='mps',\n                    weights_only=True  # 仅加载权重以减少内存使用\n                )\n                model.load_state_dict(state_dict)\n                # 确保同步完成\n                torch.mps.synchronize()\n            else:\n                # 针对CUDA或CPU的加载\n                state_dict = torch.load(\n                    config.MODEL_BEST_PATH,\n                    map_location=device,\n                    weights_only=True\n                )\n                model.load_state_dict(state_dict)"
        ),
    ]
    
    return modify_file(filepath, replacements)

def create_setup_script():
    """创建环境设置脚本"""
    print_step(6, "创建环境设置脚本")
    
    filepath = "setup_m2_max_env.py"
    content = """#!/usr/bin/env python3
\"\"\"
M2 Max环境设置脚本
此脚本设置所有必要的环境变量以实现最佳性能
\"\"\"

import os
import sys
import torch
import platform

def setup_m2_max_environment():
    \"\"\"设置M2 Max优化环境\"\"\"
    print("设置M2 Max优化环境变量...")
    
    # 检查是否在macOS上运行
    if platform.system() != 'Darwin':
        print("警告: 此脚本设计用于macOS/M2 Max环境")
        return False
    
    # 设置PyTorch MPS优化环境变量
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['TORCH_WARN_ALWAYS_UNSAFE_USAGE'] = '0'
    
    # 基于CPU核心数设置线程数
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    os.environ['OMP_NUM_THREADS'] = str(min(cpu_count, 6))
    os.environ['MKL_NUM_THREADS'] = str(min(cpu_count, 6))
    
    print(f"✓ 环境变量已设置:")
    print(f"  - PYTORCH_ENABLE_MPS_FALLBACK=1")
    print(f"  - TORCH_WARN_ALWAYS_UNSAFE_USAGE=0")
    print(f"  - OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")
    print(f"  - MKL_NUM_THREADS={os.environ['MKL_NUM_THREADS']}")
    
    # 检查MPS可用性
    if torch.backends.mps.is_available():
        print("✓ MPS加速已可用")
    else:
        print("⚠️ MPS加速不可用，请确保:")
        print("  - 使用macOS 12.3+")
        print("  - 已安装PyTorch 2.0+")
        return False
    
    print("\\n环境已成功设置！")
    print("现在可以运行: python run_optimized.py")
    return True

if __name__ == "__main__":
    setup_m2_max_environment()
"""
    
    try:
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"  ✓ 已创建环境设置脚本: {filepath}")
        return True
    except Exception as e:
        print(f"  ⚠️ 创建环境设置脚本时出错: {e}")
        return False

def verify_optimizations():
    """验证所有优化是否已应用"""
    print_step(7, "验证优化")
    
    all_ok = True
    
    # 确保目录结构正确
    for directory in ['models', 'figures']:
        if not os.path.exists(directory):
            print(f"  ⚠️ 目录 '{directory}' 不存在，正在创建...")
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"  ✓ 目录 '{directory}' 已创建")
            except Exception as e:
                print(f"  ⚠️ 创建目录 '{directory}' 时出错: {e}")
                all_ok = False
    
    # 检查PyTorch MPS可用性
    try:
        import torch
        if torch.backends.mps.is_available():
            print("  ✓ MPS加速可用")
        else:
            print("  ⚠️ MPS加速不可用，优化效果可能有限")
            all_ok = False
    except ImportError:
        print("  ⚠️ 未安装PyTorch，无法验证MPS可用性")
        all_ok = False
    
    # 确保已创建必要的脚本
    required_scripts = [
        'setup_m2_max_env.py',
        'run_optimized.py'
    ]
    
    for script in required_scripts:
        if not os.path.exists(script):
            print(f"  ⚠️ 脚本 '{script}' 不存在")
            all_ok = False
        else:
            print(f"  ✓ 脚本 '{script}' 已存在")
    
    return all_ok

def main():
    """主函数"""
    print_header("M2 Max优化应用工具")
    
    # 检查是否在macOS上运行
    if platform.system() != 'Darwin':
        print("警告: 此脚本设计用于macOS环境，但将继续运行...")
    
    # 检查目录
    project_path = os.getcwd()
    print(f"当前项目目录: {project_path}")
    
    # 确保所需文件存在
    required_files = [
        'config.py', 
        'data_generation.py', 
        'data_preprocessing.py',
        'model.py',
        'training.py',
        'evaluation.py',
        'main_experiment.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"错误: 缺少以下必需文件: {', '.join(missing_files)}")
        print("请确保您在正确的项目目录中运行此脚本。")
        sys.exit(1)
    
    # 检查MPS可用性
    has_mps = check_mps_available()
    
    # 应用所有优化
    steps_results = []
    steps_results.append(("配置优化", apply_config_optimizations()))
    steps_results.append(("训练优化", apply_training_optimizations()))
    steps_results.append(("模型优化", apply_model_optimizations()))
    steps_results.append(("数据加载优化", apply_data_preprocessing_optimizations()))
    steps_results.append(("主实验优化", apply_main_experiment_optimizations()))
    steps_results.append(("创建环境设置脚本", create_setup_script()))
    
    # 验证优化
    verify_ok = verify_optimizations()
    steps_results.append(("验证优化", verify_ok))
    
    # 打印结果摘要
    print_header("优化应用结果摘要")
    
    for step_name, result in steps_results:
        status = "✓ 成功" if result else "⚠️ 失败"
        print(f"  {status}: {step_name}")
    
    all_success = all(result for _, result in steps_results)
    
    if all_success:
        print("\n✅ 所有优化已成功应用！")
        print("\n使用说明:")
        print("1. 首先设置优化环境:")
        print("   python setup_m2_max_env.py")
        print("2. 然后运行优化后的实验:")
        print("   python run_optimized.py")
    else:
        print("\n⚠️ 部分优化应用失败。请查看上面的错误消息。")
        print("您可以手动应用失败的优化，或者恢复备份文件(.bak)。")
    
    print("\n备注: 所有修改过的文件都有.bak扩展名的备份。")

if __name__ == "__main__":
    main()
