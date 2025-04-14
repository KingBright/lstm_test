# training.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import os
import numpy as np
import config

def train_model(model, train_loader, val_loader, num_epochs=config.NUM_EPOCHS,
                learning_rate=config.LEARNING_RATE, device=None,
                early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
                scheduler_factor=config.SCHEDULER_FACTOR,
                scheduler_patience=config.SCHEDULER_PATIENCE,
                weight_decay=config.WEIGHT_DECAY,
                model_save_path=config.MODEL_BEST_PATH,
                final_model_info_path=config.MODEL_FINAL_PATH):
    """
    Trains the model with diagnostics for model output vs target and gradient norms.
    Uses ReduceLROnPlateau scheduler. Corrected gradient printing for final layer.
    """
    start_time = time.time()
    if device is None:
        if torch.backends.mps.is_available(): device = torch.device("mps"); print("使用 MPS 加速进行训练。")
        elif torch.cuda.is_available(): device = torch.device("cuda"); print("使用 CUDA 加速进行训练。")
        else: device = torch.device("cpu"); print("使用 CPU 进行训练。")
    model.to(device)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    try: total_params = sum(p.numel() for p in model.parameters() if p.requires_grad); print(f"模型可训练参数数量: {total_params:,}")
    except Exception as e: print(f"无法计算模型参数: {e}")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True, threshold=1e-4, min_lr=1e-7)
    print(f"学习率调度: 使用 ReduceLROnPlateau (factor={scheduler_factor}, patience={scheduler_patience})")

    train_loader_len = 0; val_loader_len = 0
    try:
         train_loader_len = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else len(train_loader)
         val_loader_len = len(val_loader.dataset) if val_loader and hasattr(val_loader, 'dataset') else (len(val_loader) if val_loader else 0)
         if train_loader_len == 0: print("错误: 训练数据集为空。"); return [], [], 0
         if val_loader_len == 0: print("警告: 验证数据集为空。")
    except Exception as e: print(f"错误: 无法确定数据集长度: {e}。"); return [], [], 0

    best_val_loss = float('inf'); best_epoch = 0; early_stopping_counter = 0
    train_losses = []; val_losses = []
    print(f"开始训练，最多 {num_epochs} 个周期...")
    epochs_ran = 0
    diag_print_freq = 100 # Frequency to print diagnostics

    for epoch in range(num_epochs):
        epochs_ran = epoch + 1
        epoch_start_time = time.time()
        prev_lr = optimizer.param_groups[0]['lr']

        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        train_batches = 0
        grad_accumulation_steps = 1

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # --- Forward pass ---
            try:
                if grad_accumulation_steps == 1: optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                if grad_accumulation_steps > 1: loss = loss / grad_accumulation_steps
            except Exception as e: print(f"训练前向/损失计算错误 (周期 {epoch+1}): {e}"); return train_losses, val_losses, 0
            if not torch.isfinite(loss): print(f"错误: 训练周期 {epoch+1} 损失 NaN/Inf。"); return train_losses, val_losses, 0

            # --- Backward pass ---
            try:
                loss.backward() # Calculate gradients
            except Exception as e: print(f"训练反向传播错误 (周期 {epoch+1}): {e}"); return train_losses, val_losses, 0


            # --- Diagnostics (Print Outputs, Targets, Gradients) ---
            print_diagnostics = (epoch == 0 or (epoch + 1) % 10 == 0) and batch_idx % diag_print_freq == 0
            if print_diagnostics:
                print(f"\n--- DIAGNOSTIC [Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}] ---")
                print(f"  Outputs shape: {outputs.shape}, Targets shape: {targets.shape}")
                k_print = min(getattr(config, 'OUTPUT_SEQ_LEN', 5), 5)
                # Use .item() for single loss value, ensure grad_accumulation_steps is handled if > 1
                current_loss_item = loss.item() * grad_accumulation_steps if grad_accumulation_steps > 1 else loss.item()
                print(f"  Targets (Scaled, Sample 0, Steps 0-{k_print-1}):\n{targets[0, :k_print, :].detach().cpu().numpy()}")
                print(f"  Outputs (Scaled, Sample 0, Steps 0-{k_print-1}):\n{outputs[0, :k_print, :].detach().cpu().numpy()}")
                print(f"  Loss (this batch): {current_loss_item:.6f}")

                # --- Gradient Norm Diagnostics ---
                print(f"  --- Gradient Norms (L2) ---")
                total_model_grad_norm = 0.0
                try:
                    # Iterate through named parameters to identify layers
                    for name, p in model.named_parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2).item()
                            total_model_grad_norm += param_norm ** 2
                            # Print norms for specific layers of interest
                            # Use startswith for more robustness against bidirectional prefixes etc.
                            if name.startswith('lstm1.') or name.startswith('gru1.'):
                                print(f"    {name}: {param_norm:.6e}")
                            elif name.startswith('lstm2.') or name.startswith('gru2.'):
                                print(f"    {name}: {param_norm:.6e}")
                            # *** FIXED: Check final linear layer in mlp_output_layer by index ***
                            elif name.startswith('mlp_output_layer.2.'): # Assuming final Linear is index 2 in Sequential
                                print(f"    {name} (Final Linear): {param_norm:.6e}")
                            # Optionally print other layers like mlp_input_layer or res_blocks
                            # elif name.startswith('mlp_input_layer.'):
                            #      print(f"    {name}: {param_norm:.6e}")
                            # elif name.startswith('res_block'):
                            #      print(f"    {name}: {param_norm:.6e}")

                    total_model_grad_norm = total_model_grad_norm ** 0.5
                    print(f"    Total Model Grad Norm: {total_model_grad_norm:.6e}")
                except Exception as grad_e:
                    print(f"    Error calculating gradient norms: {grad_e}")
                print(f"  ---------------------------")
                # --- END OF GRADIENT PRINTS ---
                print(f"----------------------------------------------------")


            # --- Optimizer Step ---
            try:
                # Apply gradient clipping BEFORE optimizer step
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                if (batch_idx + 1) % grad_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    optimizer.step()
                    # Zero gradients AFTER stepping if accumulating or standard training
                    optimizer.zero_grad()

            except Exception as e: print(f"优化器步骤错误 (周期 {epoch+1}): {e}"); return train_losses, val_losses, 0

            running_train_loss += current_loss_item # Use the item loss
            train_batches += 1

        if device.type == 'mps': torch.mps.synchronize()
        epoch_train_loss = running_train_loss / train_batches if train_batches > 0 else 0
        train_losses.append(epoch_train_loss)

        # --- Validation Phase ---
        model.eval(); running_val_loss = 0.0; val_batches = 0; epoch_val_loss = float('nan')
        if val_loader is not None and val_loader_len > 0:
             with torch.no_grad():
                  for inputs, targets in val_loader:
                       inputs = inputs.to(device, non_blocking=True); targets = targets.to(device, non_blocking=True)
                       try: outputs = model(inputs); loss = criterion(outputs, targets)
                       except Exception as e: print(f"验证前向/损失计算错误: {e}"); running_val_loss = float('nan'); break
                       if not torch.isfinite(loss): print(f"警告: 验证周期 {epoch+1} 损失 NaN/Inf。"); running_val_loss = float('nan'); break
                       running_val_loss += loss.item(); val_batches += 1
             if val_batches > 0 and np.isfinite(running_val_loss): epoch_val_loss = running_val_loss / val_batches
             else: epoch_val_loss = float('nan')
             val_losses.append(epoch_val_loss)
             if np.isfinite(epoch_val_loss): scheduler.step(epoch_val_loss)
             else: print(f"警告: 周期 {epoch+1} 验证损失无效，跳过 LR 调度器步骤。")
        else: val_losses.append(float('nan'))

        # --- Logging & Early Stopping ---
        epoch_time = time.time() - epoch_start_time; current_lr = optimizer.param_groups[0]['lr']
        val_loss_str = f"{epoch_val_loss:.6f}" if np.isfinite(epoch_val_loss) else "N/A"
        print(f'周期 [{epoch+1}/{num_epochs}], 训练损失: {epoch_train_loss:.6f}, 验证损失: {val_loss_str}, 耗时: {epoch_time:.2f}s, 学习率: {current_lr:.6e}')

        if np.isfinite(epoch_val_loss):
            if epoch_val_loss < best_val_loss - 1e-5:
                best_val_loss = epoch_val_loss; best_epoch = epoch + 1
                try: torch.save(model.state_dict(), model_save_path); early_stopping_counter = 0
                except Exception as e: print(f"  -> 保存最佳模型时出错: {e}")
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print(f"触发早停 (Patience: {early_stopping_patience}). 最佳周期: {best_epoch}."); break
        elif val_loader_len == 0 and epoch > 0:
             stagnation_patience = early_stopping_patience
             if epoch > stagnation_patience and len(train_losses) > stagnation_patience:
                  recent_losses = train_losses[-stagnation_patience:]
                  if all(abs(recent_losses[0] - tl) < 1e-5 for tl in recent_losses):
                       print(f"由于训练损失平稳（无验证数据），在周期 {epoch+1} 提前停止。"); break

    # --- Post-Training ---
    total_time = time.time() - start_time; print(f"\n训练完成。总耗时: {total_time:.2f}s")
    final_model_info = { 'best_epoch': best_epoch if best_epoch > 0 else epochs_ran, 'best_val_loss': best_val_loss if np.isfinite(best_val_loss) else None,'final_train_loss': epoch_train_loss if np.isfinite(epoch_train_loss) else None, 'model_type': config.MODEL_TYPE, 'final_epoch_ran': epochs_ran, 'learning_rate': learning_rate, 'weight_decay': weight_decay, 'batch_size': config.BATCH_SIZE, 'input_sequence_length': config.INPUT_SEQ_LEN, 'output_sequence_length': config.OUTPUT_SEQ_LEN, 'predict_delta': config.PREDICT_DELTA, 'use_sincos': config.USE_SINCOS_THETA, 'predict_sincos_output': config.PREDICT_SINCOS_OUTPUT }
    best_model_loaded = False
    if best_epoch > 0 and os.path.exists(model_save_path):
        print(f"加载来自周期 {best_epoch} 的最佳模型状态...");
        try:
            if device.type == 'mps': state_dict = torch.load(model_save_path, map_location='mps', weights_only=True)
            else: state_dict = torch.load(model_save_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict); print(f"最佳模型验证损失: {best_val_loss:.6f}"); final_model_info['comment'] = f"Loaded best model from epoch {best_epoch}"; best_model_loaded = True
        except Exception as e: print(f"加载最佳模型状态时出错: {e}"); print("将保存最终状态。"); final_model_info['comment'] = "Failed to load best state, saved final state."
    else:
        if best_epoch == 0 and val_loader_len > 0: print("验证损失未改善或无验证集。")
        elif best_epoch > 0: print(f"警告: 找不到最佳模型文件 {model_save_path}。")
        print("将保存训练结束时的模型状态。"); final_model_info['comment'] = "Saved final model state."
    try:
        final_model_info['model_state_dict'] = model.state_dict()
        current_params = config.get_current_model_params(); input_size_saved = 'unknown'; output_size_saved = 'unknown'
        if hasattr(model, 'input_norm') and isinstance(model.input_norm, nn.LayerNorm): input_size_saved = model.input_norm.normalized_shape[0]
        if hasattr(model, 'mlp_output_layer') and isinstance(model.mlp_output_layer, nn.Sequential) and len(model.mlp_output_layer) > 0:
             final_linear = model.mlp_output_layer[-1] # Get last layer of output sequential
             if isinstance(final_linear, nn.Linear):
                 output_features_per_step = 3 if config.PREDICT_SINCOS_OUTPUT else 2
                 if final_linear.out_features != config.OUTPUT_SEQ_LEN * output_features_per_step: print(f"Warning: Model output layer size mismatch.")
                 output_size_saved = output_features_per_step
        final_model_info.update({ 'input_size': input_size_saved, 'hidden_size': current_params.get('hidden_size', 'unknown'), 'num_layers': current_params.get('num_layers', 'unknown'), 'output_size': output_size_saved, 'dense_units': current_params.get('dense_units', 'unknown'), 'dropout_rate': current_params.get('dropout_rate', 'unknown') })
        torch.save(final_model_info, final_model_info_path); print(f"最终模型信息已保存到 {final_model_info_path}")
    except Exception as e: print(f"保存最终模型信息时出错: {e}")

    return train_losses, val_losses, best_epoch
