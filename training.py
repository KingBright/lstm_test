# training.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import os
import numpy as np # Import numpy for isnan/isfinite checks
import config # Import config for hyperparameters and paths
# model.py is not directly needed here as the model instance is passed in

def train_model(model, train_loader, val_loader, num_epochs=config.NUM_EPOCHS,
                learning_rate=config.LEARNING_RATE, device=None,
                early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
                scheduler_factor=config.SCHEDULER_FACTOR,
                scheduler_patience=config.SCHEDULER_PATIENCE,
                weight_decay=config.WEIGHT_DECAY,
                model_save_path=config.MODEL_BEST_PATH,
                final_model_info_path=config.MODEL_FINAL_PATH):
    """
    训练指定的神经网络模型。

    Args:
        model (nn.Module): 要训练的神经网络模型。
        train_loader (DataLoader): 训练集的 DataLoader。
        val_loader (DataLoader or None): 验证集的 DataLoader。可以为 None。
        num_epochs (int): 最大训练周期数。
        learning_rate (float): 初始学习率。
        device (torch.device): 训练设备 (例如 'cpu', 'cuda', 'mps')。
        early_stopping_patience (int): 早停的耐心轮数。
        scheduler_factor (float): 学习率降低因子。
        scheduler_patience (int): 学习率调度器的耐心轮数。
        weight_decay (float): 优化器的权重衰减 (L2 惩罚)。
        model_save_path (str): 保存最佳模型状态字典的路径。
        final_model_info_path (str): 保存最终模型信息（包括状态字典）的路径。

    Returns:
        tuple: (train_losses, val_losses, best_epoch) 包含训练历史。
               如果训练无法进行，则返回 ([], [], 0)。
    """
    start_time = time.time()

    # 如果未提供设备，则自动检测
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps"); print("使用 MPS 加速进行训练。")
        elif torch.cuda.is_available():
            device = torch.device("cuda"); print("使用 CUDA 加速进行训练。")
        else:
            device = torch.device("cpu"); print("使用 CPU 进行训练。")
    model.to(device)

    # 确保模型保存目录存在
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # 计算可训练参数数量
    try:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型可训练参数数量: {total_params:,}")
    except Exception as e:
        print(f"无法计算模型参数: {e}")

    # 损失函数 (MSE 适用于状态或差值预测)
    criterion = nn.MSELoss()

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 学习率调度器 (基于验证损失)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',           # 当指标停止下降时降低学习率
        factor=scheduler_factor,
        patience=scheduler_patience,
        verbose=True,
        min_lr=1e-7           # 设置一个最小学习率
    )

    # 检查 DataLoader 是否有效
    train_loader_len = 0
    val_loader_len = 0
    try:
         train_loader_len = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else len(train_loader)
         val_loader_len = len(val_loader.dataset) if val_loader and hasattr(val_loader, 'dataset') else (len(val_loader) if val_loader else 0)
         if train_loader_len == 0: print("错误: 训练数据集为空。无法训练模型。"); return [], [], 0
         if val_loader_len == 0: print("警告: 验证数据集为空或加载器为 None。将在没有验证指标的情况下进行训练。")
    except (TypeError, AttributeError) as e: print(f"错误: 无法确定数据集长度: {e}。无法可靠地训练模型。"); return [], [], 0

    # 初始化训练循环变量
    best_val_loss = float('inf')
    best_epoch = 0
    early_stopping_counter = 0
    train_losses = []
    val_losses = [] # 如果没有验证数据，则存储 NaN

    print(f"开始训练，最多 {num_epochs} 个周期...")
    # 实际运行的周期数
    epochs_ran = 0
    for epoch in range(num_epochs):
        epochs_ran = epoch + 1 # 记录实际运行到的周期
        epoch_start_time = time.time()

        # --- 训练阶段 ---
        model.train() # 设置为训练模式
        running_train_loss = 0.0
        train_batches = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if not torch.isfinite(loss): print(f"错误: 在训练周期 {epoch+1} 检测到 NaN 或 Inf 损失。停止训练。"); return train_losses, val_losses, 0
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪
            optimizer.step()
            running_train_loss += loss.item()
            train_batches += 1
        epoch_train_loss = running_train_loss / train_batches if train_batches > 0 else 0
        train_losses.append(epoch_train_loss)

        # --- 验证阶段 ---
        model.eval() # 设置为评估模式
        running_val_loss = 0.0
        val_batches = 0
        epoch_val_loss = float('nan') # 默认为 NaN

        if val_loader is not None and val_loader_len > 0:
             with torch.no_grad():
                  for inputs, targets in val_loader:
                       inputs, targets = inputs.to(device), targets.to(device)
                       outputs = model(inputs)
                       loss = criterion(outputs, targets)
                       if not torch.isfinite(loss): print(f"警告: 在验证周期 {epoch+1} 检测到 NaN 或 Inf 损失。"); running_val_loss = float('nan'); break
                       running_val_loss += loss.item()
                       val_batches += 1
             if val_batches > 0 and np.isfinite(running_val_loss): epoch_val_loss = running_val_loss / val_batches
             else: epoch_val_loss = float('nan')
             val_losses.append(epoch_val_loss)
             # 仅当验证损失有效时才更新调度器
             if np.isfinite(epoch_val_loss): scheduler.step(epoch_val_loss)
             else: print("由于验证损失无效，跳过 LR 调度器步骤。")
        else: # 如果没有验证加载器
             val_losses.append(float('nan'))

        # --- 日志记录 ---
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            val_loss_str = f"{epoch_val_loss:.6f}" if np.isfinite(epoch_val_loss) else "N/A"
            print(f'周期 [{epoch+1}/{num_epochs}], 训练损失: {epoch_train_loss:.6f}, 验证损失: {val_loss_str}, '
                  f'耗时: {epoch_time:.2f}s, 学习率: {current_lr:.6e}')

        # --- 早停和保存最佳模型 ---
        if np.isfinite(epoch_val_loss): # 仅当有有效的验证损失时执行
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch + 1
                try:
                    torch.save(model.state_dict(), model_save_path) # 保存最佳模型的状态字典
                    early_stopping_counter = 0
                    # print(f"  -> 新的最佳模型已保存到 {model_save_path} (Epoch {best_epoch}, Val Loss: {best_val_loss:.6f})") # 可以取消注释以获得更详细的日志
                except Exception as e: print(f"  -> 保存最佳模型时出错: {e}")
            else: # 如果验证损失没有改善
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print(f"基于验证损失，在周期 {epoch + 1} 触发早停。最佳周期为 {best_epoch}。")
                    break # 停止训练
        elif val_loader_len == 0 and epoch > early_stopping_patience: # 如果没有验证数据，可选地基于训练损失平稳来早停
             if epoch > early_stopping_patience + 5 and len(train_losses) > 5:
                  recent_train_losses = train_losses[-5:]
                  if all(abs(recent_train_losses[i] - recent_train_losses[i-1]) < 1e-6 for i in range(1, 5)):
                       print(f"由于训练损失平稳（无验证数据），在周期 {epoch+1} 提前停止。")
                       break

    # --- 训练后处理 ---
    total_time = time.time() - start_time
    print(f"\n训练完成。总耗时: {total_time:.2f}s")

    # 准备最终模型信息字典
    final_model_info = {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss if np.isfinite(best_val_loss) else None,
        'model_type': config.MODEL_TYPE, # 保存训练时使用的模型类型
        'final_epoch_ran': epochs_ran,   # 保存实际运行的周期数
        # 保存训练超参数以供参考
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'batch_size': config.BATCH_SIZE,
        'sequence_length': config.SEQUENCE_LENGTH
    }

    # 加载最佳模型状态（如果存在），否则使用最终状态
    if best_epoch > 0 and os.path.exists(model_save_path):
        print(f"加载来自周期 {best_epoch} 的最佳模型状态...")
        try:
            model.load_state_dict(torch.load(model_save_path))
            print(f"最佳模型验证损失: {best_val_loss:.6f}")
            final_model_info['model_state_dict'] = model.state_dict() # 保存最佳状态
        except Exception as e:
            print(f"从 {model_save_path} 加载最佳模型状态时出错: {e}")
            print("将保存训练结束时的模型状态。")
            final_model_info['model_state_dict'] = model.state_dict() # 保存最终状态
            final_model_info['comment'] = "加载最佳状态失败，保存的是最终状态。"
            best_epoch = 0 # 标记最佳状态加载失败
    else: # 如果没有找到最佳周期或文件不存在
        if best_epoch == 0 and val_loader_len > 0: print("训练期间验证损失未改善。")
        elif best_epoch > 0: print(f"警告: 找不到最佳模型文件 {model_save_path}。")
        print("将保存训练结束时的模型状态。")
        final_model_info['model_state_dict'] = model.state_dict() # 保存最终状态

    # 保存最终模型信息（包含状态字典和超参数）
    try:
        # 尝试动态获取模型架构参数
        current_params = config.get_current_model_params()
        input_size_saved = 'unknown'
        if hasattr(model, 'input_norm') and isinstance(model.input_norm, nn.LayerNorm):
             input_size_saved = model.input_norm.normalized_shape[0]
        output_size_saved = 'unknown'
        if hasattr(model, 'output_net') and isinstance(model.output_net, nn.Sequential) and len(model.output_net) > 0 and isinstance(model.output_net[-1], nn.Linear):
             output_size_saved = model.output_net[-1].out_features

        final_model_info.update({
            'input_size': input_size_saved,
            'hidden_size': current_params.get('hidden_size', 'unknown'),
            'num_layers': current_params.get('num_layers', 'unknown'),
            'output_size': output_size_saved,
            'dense_units': current_params.get('dense_units', 'unknown'),
            'dropout_rate': current_params.get('dropout_rate', 'unknown')
        })
        torch.save(final_model_info, final_model_info_path)
        print(f"最终模型信息 (包含状态字典) 已保存到 {final_model_info_path}")
    except Exception as e:
         print(f"保存最终模型信息时出错: {e}")

    # 返回训练历史和最佳周期编号
    return train_losses, val_losses, best_epoch
