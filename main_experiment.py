# main_experiment.py

import time
import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch.nn as nn
from collections import defaultdict # To collect dataframes by scenario

# Import necessary modules from the project
import config
import utils
# Import functions/classes from respective modules
from data_generation import PendulumSystem, run_simulation
# 使用改进的数据生成方法
from improved_data_generation import generate_improved_dataset, analyze_dataset_coverage
# --- VVVVVV 确认导入了新的数据准备函数 VVVVVV ---
from data_preprocessing import prepare_timesplit_seq2seq_data, create_sequences
# --- ^^^^^^ 确认导入了新的数据准备函数 ^^^^^^ ---
from model import get_model # Use factory function
from training import train_model
# Import necessary evaluation functions
from evaluation import evaluate_model, multi_step_prediction, plot_training_curves, plot_multi_step_prediction

def run_experiment():
    """
    Runs the complete workflow using long random trajectory generation,
    chronological sequence splitting, Seq2Seq model training/evaluation.
    """
    start_time = time.time()

    # --- Setup ---
    utils.setup_logging_and_warnings()
    utils.setup_chinese_font()
    
    # 优化的设备设置，包括MPS加速
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        # 设置环境变量来优化Metal性能
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        print(f"使用 MPS 加速 (M2 Max)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 CUDA 加速")
    else:
        device = torch.device("cpu")
        print(f"使用 CPU 计算")
        
    # 设置随机种子以确保可重复性
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # --- Step 1: System Definition ---
    pendulum = PendulumSystem(m=config.PENDULUM_MASS, L=config.PENDULUM_LENGTH,
                              g=config.GRAVITY, c=config.DAMPING_COEFF)
    print("步骤1: 已创建单摆系统")

    # --- Step 2: 生成优化的训练数据 ---
    print("步骤2: 生成优化的训练数据...")
    gen_start_time = time.time()
    df_all = None # 初始化df_all
    simulation_boundaries = [] # 存储模拟边界信息
    total_points = 0
    
    # 检查是否存在已合并的数据文件且不强制重新生成
    if os.path.exists(config.COMBINED_DATA_FILE) and not config.FORCE_REGENERATE_DATA:
        print(f"加载已有的合并数据文件: {config.COMBINED_DATA_FILE}...")
        try:
            df_all = pd.read_csv(config.COMBINED_DATA_FILE)
            total_points = len(df_all)
            print(f"已加载 {total_points} 数据点.")
            
            # 计算生成的序列数估计值
            seq_length = config.INPUT_SEQ_LEN + config.OUTPUT_SEQ_LEN
            estimated_sequences = max(0, total_points - seq_length + 1)
            print(f"估计可创建的序列数: {estimated_sequences}")
            
            if estimated_sequences < config.TARGET_SEQUENCES:
                print(f"警告: 估计序列数 ({estimated_sequences}) 低于目标序列数 ({config.TARGET_SEQUENCES}).")
                print("将继续生成更多数据以达到目标...")
                # 使用改进的数据生成方法
                print("使用改进的数据生成方法添加额外数据...")
                additional_df, additional_boundaries = generate_improved_dataset(
                    target_sequences=config.TARGET_SEQUENCES - estimated_sequences,
                    dt=config.DT,
                    t_span=config.T_SPAN,
                    output_file=config.COMBINED_DATA_FILE + ".additional"
                )
                
                if additional_df is not None and not additional_df.empty:
                    # 更新模拟边界
                    if len(simulation_boundaries) > 0:
                        # 如果已有边界信息，需要调整新边界
                        offset = total_points
                        adjusted_boundaries = [b + offset for b in additional_boundaries]
                        simulation_boundaries.extend(adjusted_boundaries)
                    else:
                        # 首次添加边界信息
                        simulation_boundaries = additional_boundaries.copy()
                    
                    # 合并数据
                    df_all = pd.concat([df_all, additional_df], ignore_index=True)
                    total_points = len(df_all)
                    
                    # 保存合并后的数据
                    try:
                        df_all.to_csv(config.COMBINED_DATA_FILE, index=False)
                        print(f"合并后的数据已保存到 {config.COMBINED_DATA_FILE}")
                    except Exception as e:
                        print(f"保存合并数据文件时出错: {e}")
            else:
                print(f"已有数据足够生成目标序列数量.")
        except Exception as e:
            print(f"加载合并数据文件时出错: {e}. 将使用改进的方法重新生成.")
            df_all = None
            config.FORCE_REGENERATE_DATA = True  # 强制重新生成
    
    # 如果需要生成新数据
    if df_all is None or config.FORCE_REGENERATE_DATA:
        print("使用改进的数据生成方法创建全新训练数据...")
        df_all, simulation_boundaries = generate_improved_dataset(
            target_sequences=config.TARGET_SEQUENCES,
            dt=config.DT,
            t_span=config.T_SPAN,
            output_file=config.COMBINED_DATA_FILE
        )
        
        # 分析数据集的覆盖情况
        if df_all is not None and not df_all.empty:
            print("分析生成数据的覆盖情况...")
            coverage, uniformity = analyze_dataset_coverage(df_all, config.FIGURES_DIR)
            print(f"状态空间覆盖率: {coverage:.2f}%, 均匀度: {uniformity:.2f}%")
        else:
            print("错误: 数据生成失败，无法继续.")
            return

    gen_time = time.time() - gen_start_time
    print(f"数据生成/加载完成，耗时: {gen_time:.2f}秒")
    if df_all is None or df_all.empty: 
        print("数据加载/生成失败，无法继续."); 
        return

    # --- Physics Validation Step REMOVED ---

    # --- Step 3: Prepare DataLoaders using Chronological Split ---
    print("\n步骤3: 创建序列并按时序分割训练/验证集...")
    data_prep_start_time = time.time()
    
    # 使用生成的模拟边界信息，不需要重新计算
    print(f"找到 {len(simulation_boundaries)+1 if simulation_boundaries else 0} 个独立模拟，使用模拟边界确保序列不跨越不同模拟数据")
    
    # --- VVVVVV 调用新的时序分割数据准备函数，传入模拟边界 VVVVVV ---
    data_loaders_tuple = prepare_timesplit_seq2seq_data(
        df_all, # Pass the combined dataframe
        simulation_boundaries=simulation_boundaries,  # 传入模拟边界
        input_sequence_length=config.INPUT_SEQ_LEN,
        output_sequence_length=config.OUTPUT_SEQ_LEN,
        val_split_ratio=config.VALIDATION_SPLIT, # Pass the split ratio
        seed=config.SEED
        # Scaler types/paths are handled inside using config
    )
    # --- ^^^^^^ 调用新的时序分割数据准备函数 ^^^^^^ ---

    if data_loaders_tuple is None or data_loaders_tuple[0] is None: print("Error: Failed to create datasets and loaders."); return
    train_loader, val_loader, input_scaler, target_scaler = data_loaders_tuple
    data_prep_time = time.time() - data_prep_start_time
    print(f"数据准备完成，耗时: {data_prep_time:.2f}秒")
    train_loader_len_check = len(train_loader) if train_loader else 0; val_loader_len_check = len(val_loader) if val_loader else 0
    print(f"Train loader batches: {train_loader_len_check}, Val loader batches: {val_loader_len_check}")
    if train_loader_len_check == 0: print("Error: Training loader is empty."); return

    # --- Step 4: Model Definition ---
    print("\n步骤4: 创建神经网络模型...")
    try:
        # 确定正确的输入和输出大小
        input_size = 4 if config.USE_SINCOS_THETA else 3
        output_size = 2  # theta和theta_dot
        
        # 从scaler获取确切的特征数量（更加可靠）
        if hasattr(input_scaler, 'n_features_in_'):
            if input_scaler.n_features_in_ != input_size:
                print(f"  注意: 使用来自scaler的input_size = {input_scaler.n_features_in_} (原预期值: {input_size})")
                input_size = input_scaler.n_features_in_
            else:
                print(f"  确认: input_size = {input_size}")
                
        if hasattr(target_scaler, 'n_features_in_'):
            if target_scaler.n_features_in_ != output_size:
                print(f"  注意: 使用来自scaler的output_size = {target_scaler.n_features_in_} (原预期值: {output_size})")
                output_size = target_scaler.n_features_in_
            else:
                print(f"  确认: output_size = {output_size}")
    except AttributeError as e:
        print(f"错误: Scalers无效: {e}")
        return
        
    try:
        # 创建模型
        model = get_model(model_type=config.MODEL_TYPE, input_size=input_size, output_size=output_size)
        print(f"模型创建成功! 类型: {config.MODEL_TYPE}")
        print(model)
    except Exception as e:
        print(f"错误: 创建模型失败: {e}")
        return


    # --- Step 5: Model Training ---
    # ... (Model training call remains the same) ...
    print("\n步骤5: 开始训练模型...")
    train_start_time = time.time(); train_losses, val_losses, best_epoch = train_model(model, train_loader, val_loader, device=device, model_save_path=config.MODEL_BEST_PATH, final_model_info_path=config.MODEL_FINAL_PATH); train_time = time.time() - train_start_time
    print(f"模型训练完成，耗时: {train_time:.2f}秒"); plot_training_curves(train_losses, val_losses, config.FIGURES_DIR); print("训练曲线已绘制并保存")
    if best_epoch == 0: print("Warning: Training did not yield improvement.")

    # --- Step 6: Model Evaluation (Simplified for Seq2Seq) ---
    # ... (Evaluation logic remains the same, using a regenerated segment) ...
    print("\n步骤6: 开始在验证集上评估模型...")
    eval_start_time = time.time()
    
    # 优化的模型加载
    print("准备模型用于评估...")
    
    # 确保模型在正确的设备上
    model.to(device)
    
    # 尝试加载最佳模型 (如果存在)
    if os.path.exists(config.MODEL_BEST_PATH) and best_epoch > 0:
        try:
            # 针对M2 Max的优化加载
            if device.type == 'mps':
                # 使用map_location确保正确加载到MPS设备
                state_dict = torch.load(
                    config.MODEL_BEST_PATH, 
                    map_location='mps',
                    weights_only=True  # 仅加载权重以减少内存使用
                )
                model.load_state_dict(state_dict)
                # 确保同步完成
                torch.mps.synchronize()
            else:
                # 针对CUDA或CPU的加载
                state_dict = torch.load(
                    config.MODEL_BEST_PATH,
                    map_location=device,
                    weights_only=True
                )
                model.load_state_dict(state_dict)
                
            print(f"已加载最佳模型 (来自周期 {best_epoch})")
        except Exception as e:
            print(f"警告: 加载最佳模型状态失败。错误: {e}")
            print("将使用训练结束时的模型状态进行评估。")
    else:
        print("使用训练结束时的模型状态进行评估。")
    criterion = nn.MSELoss(); avg_val_loss_final = evaluate_model(model, val_loader, criterion, device) if val_loader else float('nan')
    if np.isfinite(avg_val_loss_final): print(f"最终模型在验证集上的单步 MSE: {avg_val_loss_final:.6f}") # Note: This is Seq2Seq MSE now
    else: print("无法计算验证集单步 MSE。")
    print("\n运行多步预测（使用新生成的随机片段进行检查）...")
    predicted_states, true_states, physics_predictions = None, None, None; model_mse = np.nan
    try:
        # 创建一个用于评估的单摆系统实例
        print("\n生成用于多步预测评估的随机轨迹...")
        pendulum_eval = PendulumSystem()
        
        # 使用一个随机初始条件
        eval_x0 = [np.random.uniform(*config.THETA_RANGE), np.random.uniform(*config.THETA_DOT_RANGE)]
        print(f"随机初始条件: θ = {eval_x0[0]:.3f} rad, θ̇ = {eval_x0[1]:.3f} rad/s")
        
        # 设置评估数据的持续时间
        eval_duration = 25  # 稍长一些以确保有足够的数据点
        
        # 生成随机力矩轨迹
        print(f"生成评估数据段 (持续时间: {eval_duration}s, 力矩类型: {config.TORQUE_TYPE})...")
        df_eval_segment = generate_simulation_data(
            pendulum_eval, 
            t_span=(0, eval_duration), 
            dt=config.DT, 
            x0=eval_x0, 
            torque_type=config.TORQUE_TYPE
        )
        
        # 检查数据长度是否足够
        min_required_len = config.INPUT_SEQ_LEN + config.MIN_PREDICTION_STEPS
        if not df_eval_segment.empty and len(df_eval_segment) >= min_required_len:
            print(f"成功生成评估数据: {len(df_eval_segment)} 个时间点")
            
            # 准备评估数据
            eval_data_values = df_eval_segment[['theta', 'theta_dot', 'tau']].values
            use_sincos_eval = config.USE_SINCOS_THETA
            
            # 创建模型输入序列
            X_eval, _ = create_sequences(
                eval_data_values, 
                simulation_boundaries=None,  # 单一模拟数据，不需要边界信息
                input_seq_len=config.INPUT_SEQ_LEN, 
                output_seq_len=config.OUTPUT_SEQ_LEN, 
                use_sincos=use_sincos_eval
            )
            
            if len(X_eval) > 0:
                # 检查特征数量匹配
                if input_scaler.n_features_in_ != X_eval.shape[2]:
                    raise ValueError(f"输入缩放器/数据特征不匹配: {input_scaler.n_features_in_} vs {X_eval.shape[2]}")
                
                # 缩放输入序列
                X_eval_reshaped = X_eval.reshape(-1, X_eval.shape[2])
                X_eval_scaled = input_scaler.transform(X_eval_reshaped).reshape(X_eval.shape)
                initial_sequence_eval = X_eval_scaled[0]
                
                # 准备用于预测的DataFrame切片
                df_for_pred_eval = df_eval_segment.iloc[config.INPUT_SEQ_LEN:].reset_index(drop=True)
                
                # 计算并限制预测步数
                prediction_steps_eval = len(df_for_pred_eval)
                prediction_steps_eval = min(prediction_steps_eval, 1000)  # 限制最大步数
                
                if prediction_steps_eval >= config.MIN_PREDICTION_STEPS:
                    print(f"\n执行 {prediction_steps_eval} 步预测...")
                    
                    # 执行多步预测
                    predicted_states, true_states = multi_step_prediction(
                        model, 
                        initial_sequence_eval, 
                        df_for_pred_eval, 
                        config.INPUT_SEQ_LEN, 
                        config.OUTPUT_SEQ_LEN, 
                        prediction_steps_eval, 
                        input_scaler, 
                        target_scaler, 
                        device
                    )
                    
                    if len(predicted_states) > 0:
                        # 计算模型MSE
                        model_mse = np.mean((predicted_states - true_states)**2)
                        print(f"多步预测整体MSE: {model_mse:.6f}")
                        
                        # 生成物理模型比较
                        print("进行物理模型模拟以用于比较...")
                        physics_x0 = df_eval_segment.iloc[config.INPUT_SEQ_LEN][['theta', 'theta_dot']].values
                        physics_time_eval = df_for_pred_eval['time'].iloc[:prediction_steps_eval].values
                        
                        physics_predictions = None
                        if len(physics_time_eval) > 0:
                            physics_t_span = (physics_time_eval[0], physics_time_eval[-1])
                            physics_tau_values = df_for_pred_eval['tau'].iloc[:prediction_steps_eval].values
                            
                            physics_time, physics_theta, physics_theta_dot = run_simulation(
                                pendulum_eval, 
                                physics_t_span, 
                                config.DT, 
                                physics_x0, 
                                physics_tau_values, 
                                t_eval=physics_time_eval
                            )
                            
                            if len(physics_time) == len(physics_time_eval):
                                physics_predictions = np.stack([physics_theta, physics_theta_dot], axis=1)
                                
                                # 确保物理预测和神经网络预测长度匹配
                                actual_pred_steps = len(predicted_states)
                                actual_true_steps = len(true_states)
                                actual_physics_steps = len(physics_predictions)
                                
                                # 找到三者共同的最小长度
                                min_length = min(actual_pred_steps, actual_true_steps, actual_physics_steps)
                                
                                if min_length > 0:
                                    # 截取为相同长度
                                    predicted_states_trimmed = predicted_states[:min_length]
                                    true_states_trimmed = true_states[:min_length]
                                    physics_predictions_trimmed = physics_predictions[:min_length]
                                    
                                    # 计算模型MSE (使用截取后的数据)
                                    model_mse = np.mean((predicted_states_trimmed - true_states_trimmed)**2)
                                    physics_mse = np.mean((physics_predictions_trimmed - true_states_trimmed)**2)
                                    
                                    print(f"最终使用 {min_length} 步数据计算指标")
                                    print(f"神经网络模型MSE: {model_mse:.6f}")
                                    print(f"物理模型MSE: {physics_mse:.6f}")
                                    
                                    # 更新变量以便绘图时使用
                                    true_states = true_states_trimmed
                                    predicted_states = predicted_states_trimmed
                                    physics_predictions = physics_predictions_trimmed
                                    physics_time_eval = physics_time_eval[:min_length]
                                else:
                                    print("警告: 没有足够的预测步数进行比较")
                                    physics_predictions = None
                        
                        # 绘制多步预测结果
                        print("生成详细预测图表...")
                        plot_filename = f"multistep_eval_{config.MODEL_TYPE}"
                        
                        # 确保所有数据长度一致
                        if physics_predictions is not None:
                            if len(physics_time_eval) != len(true_states) or len(physics_time_eval) != len(predicted_states) or len(physics_time_eval) != len(physics_predictions):
                                min_len = min(len(physics_time_eval), len(true_states), len(predicted_states), len(physics_predictions))
                                physics_time_eval = physics_time_eval[:min_len]
                                true_states = true_states[:min_len]
                                predicted_states = predicted_states[:min_len]
                                physics_predictions = physics_predictions[:min_len]
                                print(f"所有数据已调整为共同的长度: {min_len}")
                        
                        plot_multi_step_prediction(
                            physics_time_eval, 
                            true_states, 
                            predicted_states, 
                            physics_predictions, 
                            config.MODEL_TYPE, 
                            config.FIGURES_DIR, 
                            filename_base=plot_filename
                        )
                        print(f"多步预测图表已保存到 {config.FIGURES_DIR}/{plot_filename}.png")
                    else:
                        print("警告: 多步预测未返回任何结果。")
                else:
                    print(f"评估段可用步数 ({prediction_steps_eval}) 少于要求的最小步数 ({config.MIN_PREDICTION_STEPS})。")
            else:
                print("无法为评估段创建序列数据。")
        else:
            print(f"无法生成足够长的用于多步预测评估的数据段 (需要 >= {min_required_len} 点, 实际生成 {len(df_eval_segment)} 点)。")
    except Exception as eval_msp_e:
        print(f"执行多步预测评估时出错: {eval_msp_e}")
        import traceback
        traceback.print_exc()
    eval_time = time.time() - eval_start_time; print(f"模型评估完成，耗时: {eval_time:.2f}秒")


    # --- Final Summary ---
    # ... (Summary printing remains the same) ...
    total_time = time.time() - start_time; print(f"\n项目执行完毕！总耗时: {total_time:.2f}秒") # ... etc ...
    # ... (Print performance and model summaries) ...


if __name__ == "__main__":
    run_experiment()
