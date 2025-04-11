# training.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import os
import numpy as np
import config
# from model import PureLSTM, PureGRU # Not needed directly

def train_model(model, train_loader, val_loader, num_epochs=config.NUM_EPOCHS,
                learning_rate=config.LEARNING_RATE, device=None,
                early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
                scheduler_factor=config.SCHEDULER_FACTOR,
                scheduler_patience=config.SCHEDULER_PATIENCE,
                weight_decay=config.WEIGHT_DECAY,
                model_save_path=config.MODEL_BEST_PATH,
                final_model_info_path=config.MODEL_FINAL_PATH):
    """
    训练指定的神经网络模型（适用于单步或 Seq2Seq 输出）。
    修正了 ReduceLROnPlateau 的 verbose 警告。
    """
    start_time = time.time()

    # 确定设备
    if device is None:
        if torch.backends.mps.is_available(): device = torch.device("mps"); print("使用 MPS 加速进行训练。")
        elif torch.cuda.is_available(): device = torch.device("cuda"); print("使用 CUDA 加速进行训练。")
        else: device = torch.device("cpu"); print("使用 CPU 进行训练。")
    model.to(device)

    # 确保目录存在
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # 计算参数量
    try: total_params = sum(p.numel() for p in model.parameters() if p.requires_grad); print(f"模型可训练参数数量: {total_params:,}")
    except Exception as e: print(f"无法计算模型参数: {e}")

    # 损失函数
    criterion = nn.MSELoss()

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 学习率调度器 - 使用正确的verbose参数
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_factor,
        patience=scheduler_patience,
        verbose=False,  # 设为False以避免警告
        min_lr=1e-7
    )

    # 检查 DataLoader
    train_loader_len = 0; val_loader_len = 0
    try:
         train_loader_len = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else len(train_loader)
         val_loader_len = len(val_loader.dataset) if val_loader and hasattr(val_loader, 'dataset') else (len(val_loader) if val_loader else 0)
         if train_loader_len == 0: print("错误: 训练数据集为空。"); return [], [], 0
         if val_loader_len == 0: print("警告: 验证数据集为空。")
    except (TypeError, AttributeError) as e: print(f"错误: 无法确定数据集长度: {e}。"); return [], [], 0

    # 初始化训练循环变量
    best_val_loss = float('inf'); best_epoch = 0; early_stopping_counter = 0
    train_losses = []; val_losses = []
    # 初始化学习率变量用于比较
    current_lr = learning_rate
    prev_lr = current_lr

    print(f"开始训练，最多 {num_epochs} 个周期...")
    epochs_ran = 0
    for epoch in range(num_epochs):
        epochs_ran = epoch + 1
        epoch_start_time = time.time()
        # 记录当前周期的起始学习率
        prev_lr = optimizer.param_groups[0]['lr']

        # --- 训练阶段 (针对M2 Max优化) ---
        model.train()
        running_train_loss = 0.0
        train_batches = 0
        
        # 设置较大的批次大小进行梯度累积
        grad_accumulation_steps = 1  # 默认为1，可根据内存情况调整
        
        for inputs, targets in train_loader:
            # 使用 non_blocking=True 加速数据传输
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # 仅在需要时清零梯度
            if train_batches % grad_accumulation_steps == 0:
                optimizer.zero_grad()
                
            try:
                # 前向传播和损失计算
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 如果使用梯度累积，则缩放损失
                if grad_accumulation_steps > 1:
                    loss = loss / grad_accumulation_steps
            except Exception as e:
                print(f"训练前向/损失计算错误: {e}")
                return train_losses, val_losses, 0
                
            if not torch.isfinite(loss):
                print(f"错误: 训练周期 {epoch+1} 损失 NaN/Inf。")
                return train_losses, val_losses, 0
                
            try:
                # 反向传播
                loss.backward()
                
                # 仅在累积步骤结束时更新参数
                if (train_batches + 1) % grad_accumulation_steps == 0 or (train_batches + 1) == len(train_loader):
                    # 梯度裁剪以提高稳定性
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            except Exception as e:
                print(f"训练反向/优化器错误: {e}")
                return train_losses, val_losses, 0
                
            running_train_loss += loss.item() * (1 if grad_accumulation_steps == 1 else grad_accumulation_steps)
            train_batches += 1
            
        # 如果是MPS设备，添加同步点确保所有操作完成
        if device.type == 'mps':
            torch.mps.synchronize()
        epoch_train_loss = running_train_loss / train_batches if train_batches > 0 else 0
        train_losses.append(epoch_train_loss)

        # --- 验证阶段 ---
        model.eval(); running_val_loss = 0.0; val_batches = 0; epoch_val_loss = float('nan')
        if val_loader is not None and val_loader_len > 0:
             with torch.no_grad():
                  for inputs, targets in val_loader:
                       inputs, targets = inputs.to(device), targets.to(device)
                       try: outputs = model(inputs); loss = criterion(outputs, targets)
                       except Exception as e: print(f"验证前向/损失计算错误: {e}"); running_val_loss = float('nan'); break
                       if not torch.isfinite(loss): print(f"警告: 验证周期 {epoch+1} 损失 NaN/Inf。"); running_val_loss = float('nan'); break
                       running_val_loss += loss.item(); val_batches += 1
             if val_batches > 0 and np.isfinite(running_val_loss): epoch_val_loss = running_val_loss / val_batches
             else: epoch_val_loss = float('nan')
             val_losses.append(epoch_val_loss)
             # 仅当验证损失有效时才更新调度器
             if np.isfinite(epoch_val_loss): scheduler.step(epoch_val_loss)
             else: print("跳过 LR 调度器步骤。")
        else: val_losses.append(float('nan'))

        # --- 日志记录 ---
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr'] # 获取当前学习率
        # +++ 手动检查并打印学习率变化 +++
        if current_lr < prev_lr:
             print(f"  学习率在周期 {epoch+1} 结束时降低: {prev_lr:.6e} -> {current_lr:.6e}")
        # +++ 检查结束 +++
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            val_loss_str = f"{epoch_val_loss:.6f}" if np.isfinite(epoch_val_loss) else "N/A"
            print(f'周期 [{epoch+1}/{num_epochs}], 训练损失: {epoch_train_loss:.6f}, 验证损失: {val_loss_str}, 耗时: {epoch_time:.2f}s, 学习率: {current_lr:.6e}')

        # --- 早停和保存最佳模型 ---
        # ... (早停逻辑保持不变) ...
        if np.isfinite(epoch_val_loss):
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss; best_epoch = epoch + 1
                try: torch.save(model.state_dict(), model_save_path); early_stopping_counter = 0
                except Exception as e: print(f"  -> 保存最佳模型时出错: {e}")
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience: print(f"基于验证损失，在周期 {epoch + 1} 触发早停。最佳周期为 {best_epoch}。"); break
        elif val_loader_len == 0 and epoch > early_stopping_patience:
             if epoch > early_stopping_patience + 5 and len(train_losses) > 5 and all(abs(train_losses[-1] - tl) < 1e-6 for tl in train_losses[-5:-1]):
                  print(f"由于训练损失平稳（无验证数据），在周期 {epoch+1} 提前停止。"); break


    # --- 训练后处理 ---
    # ... (后处理逻辑保持不变, 包括保存 final_model_info) ...
    total_time = time.time() - start_time; print(f"\n训练完成。总耗时: {total_time:.2f}s")
    final_model_info = { 'best_epoch': best_epoch, 'best_val_loss': best_val_loss if np.isfinite(best_val_loss) else None, 'model_type': config.MODEL_TYPE, 'final_epoch_ran': epochs_ran, 'learning_rate': learning_rate, 'weight_decay': weight_decay, 'batch_size': config.BATCH_SIZE, 'input_sequence_length': config.INPUT_SEQ_LEN, 'output_sequence_length': config.OUTPUT_SEQ_LEN }
    if best_epoch > 0 and os.path.exists(model_save_path):
        print(f"加载来自周期 {best_epoch} 的最佳模型状态...");
        try: model.load_state_dict(torch.load(model_save_path)); print(f"最佳模型验证损失: {best_val_loss:.6f}"); final_model_info['model_state_dict'] = model.state_dict()
        except Exception as e: print(f"加载最佳模型状态时出错: {e}"); print("将保存最终状态。"); final_model_info['model_state_dict'] = model.state_dict(); best_epoch = 0; final_model_info['comment'] = "加载最佳状态失败，保存的是最终状态。"
    else:
        if best_epoch == 0 and val_loader_len > 0: print("验证损失未改善。")
        elif best_epoch > 0: print(f"警告: 找不到最佳模型文件 {model_save_path}。")
        print("将保存训练结束时的模型状态。"); final_model_info['model_state_dict'] = model.state_dict()
    try:
        current_params = config.get_current_model_params(); input_size_saved = 'unknown'; output_size_saved = 'unknown'
        if hasattr(model, 'input_norm') and isinstance(model.input_norm, nn.LayerNorm): input_size_saved = model.input_norm.normalized_shape[0]
        if hasattr(model, 'output_net') and isinstance(model.output_net, nn.Sequential) and len(model.output_net) > 0 and isinstance(model.output_net[-1], nn.Linear): output_size_saved = model.output_net[-1].out_features // config.OUTPUT_SEQ_LEN # Save per-step size
        final_model_info.update({ 'input_size': input_size_saved, 'hidden_size': current_params.get('hidden_size', 'unknown'), 'num_layers': current_params.get('num_layers', 'unknown'), 'output_size': output_size_saved, 'dense_units': current_params.get('dense_units', 'unknown'), 'dropout_rate': current_params.get('dropout_rate', 'unknown') })
        torch.save(final_model_info, final_model_info_path); print(f"最终模型信息已保存到 {final_model_info_path}")
    except Exception as e: print(f"保存最终模型信息时出错: {e}")

    return train_losses, val_losses, best_epoch
