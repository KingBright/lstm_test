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
# *** FIX: Added generate_simulation_data to the import ***
from data_generation import PendulumSystem, run_simulation, generate_simulation_data
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

            # 尝试加载边界信息 (如果存在)
            boundaries_file = config.COMBINED_DATA_FILE.replace('.csv', '_boundaries.npy')
            if os.path.exists(boundaries_file):
                simulation_boundaries = list(np.load(boundaries_file))
                print(f"已加载 {len(simulation_boundaries)} 个模拟边界。")
            else:
                print("警告: 未找到边界信息文件，将假定数据为单一模拟处理。")
                simulation_boundaries = None # 或者 [0, total_points] 如果确定是多段合并的

            # 检查序列数是否足够 (可选)
            # seq_length = config.INPUT_SEQ_LEN + config.OUTPUT_SEQ_LEN
            # estimated_sequences = max(0, total_points - (len(simulation_boundaries) + 1) * (seq_length -1) if simulation_boundaries else total_points - seq_length + 1)
            # print(f"估计可创建的序列数: {estimated_sequences}")
            # if estimated_sequences < config.TARGET_SEQUENCES:
            #    print("警告: 现有数据可能不足以生成目标序列数。")

        except Exception as e:
            print(f"加载合并数据文件或边界文件时出错: {e}. 将使用改进的方法重新生成.")
            df_all = None
            simulation_boundaries = []
            config.FORCE_REGENERATE_DATA = True  # 强制重新生成
    else:
        print(f"未找到数据文件 {config.COMBINED_DATA_FILE} 或已设置强制重新生成。")
        config.FORCE_REGENERATE_DATA = True # 确保设置为 True

    # 如果需要生成新数据
    if df_all is None or config.FORCE_REGENERATE_DATA:
        print("使用改进的数据生成方法创建全新训练数据...")
        df_all, simulation_boundaries = generate_improved_dataset(
            target_sequences=config.TARGET_SEQUENCES,
            dt=config.DT,
            t_span=config.T_SPAN,
            output_file=config.COMBINED_DATA_FILE
        )

        # 保存边界信息
        if simulation_boundaries:
             boundaries_file = config.COMBINED_DATA_FILE.replace('.csv', '_boundaries.npy')
             try:
                 np.save(boundaries_file, np.array(simulation_boundaries))
                 print(f"模拟边界信息已保存到: {boundaries_file}")
             except Exception as e:
                 print(f"保存边界信息时出错: {e}")


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

    # --- Step 3: Prepare DataLoaders using Chronological Split ---
    print("\n步骤3: 创建序列并按时序分割训练/验证集...")
    data_prep_start_time = time.time()

    # 使用加载或生成的模拟边界信息
    if simulation_boundaries:
        print(f"使用 {len(simulation_boundaries)+1} 个独立模拟的边界信息。")
    else:
        print("未提供模拟边界信息，将数据视为单一连续序列处理。")


    data_loaders_tuple = prepare_timesplit_seq2seq_data(
        df_all, # Pass the combined dataframe
        simulation_boundaries=simulation_boundaries,  # 传入模拟边界
        input_sequence_length=config.INPUT_SEQ_LEN,
        output_sequence_length=config.OUTPUT_SEQ_LEN,
        val_split_ratio=config.VALIDATION_SPLIT, # Pass the split ratio
        seed=config.SEED
    )

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
        input_size = 4 if config.USE_SINCOS_THETA else 3
        output_size = 2 # theta, theta_dot
        if hasattr(input_scaler, 'n_features_in_'): input_size = input_scaler.n_features_in_
        if hasattr(target_scaler, 'n_features_in_'): output_size = target_scaler.n_features_in_
        print(f"  确认: input_size={input_size}, output_size={output_size}")
        model = get_model(model_type=config.MODEL_TYPE, input_size=input_size, output_size=output_size)
        print(f"模型创建成功! 类型: {config.MODEL_TYPE}")
        # print(model) # Optionally print model structure
    except Exception as e:
        print(f"错误: 创建模型失败: {e}")
        import traceback
        traceback.print_exc()
        return


    # --- Step 5: Model Training ---
    print("\n步骤5: 开始训练模型...")
    train_start_time = time.time()
    train_losses, val_losses, best_epoch = train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        model_save_path=config.MODEL_BEST_PATH,
        final_model_info_path=config.MODEL_FINAL_PATH
    )
    train_time = time.time() - train_start_time
    print(f"模型训练完成，耗时: {train_time:.2f}秒")
    plot_training_curves(train_losses, val_losses, config.FIGURES_DIR)
    print("训练曲线已绘制并保存")
    if best_epoch == 0: print("Warning: Training did not yield improvement based on validation loss.")

    # --- Step 6: Model Evaluation ---
    print("\n步骤6: 开始在验证集上评估模型...")
    eval_start_time = time.time()
    print("准备模型用于评估...")
    model.to(device) # Ensure model is on the correct device

    # Load the best model state if it exists and training improved
    if os.path.exists(config.MODEL_BEST_PATH) and best_epoch > 0:
        try:
            print(f"加载最佳模型状态 (来自周期 {best_epoch})...")
            if device.type == 'mps':
                state_dict = torch.load(config.MODEL_BEST_PATH, map_location='mps', weights_only=True)
                model.load_state_dict(state_dict)
                torch.mps.synchronize()
            else:
                state_dict = torch.load(config.MODEL_BEST_PATH, map_location=device, weights_only=True)
                model.load_state_dict(state_dict)
            print("最佳模型状态加载成功。")
        except Exception as e:
            print(f"警告: 加载最佳模型状态失败。错误: {e}")
            print("将使用训练结束时的模型状态进行评估。")
    else:
        print("使用训练结束时的模型状态进行评估。")

    # Evaluate single-step prediction on validation set
    criterion = nn.MSELoss()
    avg_val_loss_final = evaluate_model(model, val_loader, criterion, device) if val_loader else float('nan')
    if np.isfinite(avg_val_loss_final): print(f"最终模型在验证集上的单步 MSE: {avg_val_loss_final:.6f}")
    else: print("无法计算验证集单步 MSE (可能验证集为空)。")

    # Perform multi-step prediction evaluation on a newly generated segment
    print("\n运行多步预测（使用新生成的随机片段进行检查）...")
    predicted_states, true_states, physics_predictions = None, None, None
    model_mse = np.nan
    try:
        print("生成用于多步预测评估的随机轨迹...")
        pendulum_eval = PendulumSystem()
        eval_x0 = [np.random.uniform(*config.THETA_RANGE), np.random.uniform(*config.THETA_DOT_RANGE)]
        print(f"随机初始条件: θ = {eval_x0[0]:.3f} rad, θ̇ = {eval_x0[1]:.3f} rad/s")
        eval_duration = 25.0
        print(f"生成评估数据段 (持续时间: {eval_duration}s, 力矩类型: {config.TORQUE_TYPE})...")

        # *** This is where the error occurred ***
        # Ensure generate_simulation_data is imported correctly (fixed above)
        df_eval_segment = generate_simulation_data(
            pendulum_eval,
            t_span=(0, eval_duration),
            dt=config.DT,
            x0=eval_x0,
            torque_type=config.TORQUE_TYPE # Use the config torque type for consistency
        )

        min_required_len = config.INPUT_SEQ_LEN + config.MIN_PREDICTION_STEPS
        if not df_eval_segment.empty and len(df_eval_segment) >= min_required_len:
            print(f"成功生成评估数据: {len(df_eval_segment)} 个时间点")
            eval_data_values = df_eval_segment[['theta', 'theta_dot', 'tau']].values
            use_sincos_eval = config.USE_SINCOS_THETA

            X_eval, _ = create_sequences(
                eval_data_values,
                simulation_boundaries=None, # Single simulation segment
                input_seq_len=config.INPUT_SEQ_LEN,
                output_seq_len=config.OUTPUT_SEQ_LEN,
                use_sincos=use_sincos_eval
            )

            if len(X_eval) > 0:
                if input_scaler.n_features_in_ != X_eval.shape[2]:
                    raise ValueError(f"评估数据特征不匹配: 缩放器={input_scaler.n_features_in_}, 数据={X_eval.shape[2]}")

                X_eval_scaled = input_scaler.transform(X_eval.reshape(-1, X_eval.shape[2])).reshape(X_eval.shape)
                initial_sequence_eval = X_eval_scaled[0]
                df_for_pred_eval = df_eval_segment.iloc[config.INPUT_SEQ_LEN:].reset_index(drop=True)
                prediction_steps_eval = min(len(df_for_pred_eval), 1000) # Limit prediction steps

                if prediction_steps_eval >= config.MIN_PREDICTION_STEPS:
                    print(f"\n执行 {prediction_steps_eval} 步预测...")
                    predicted_states, true_states = multi_step_prediction(
                        model, initial_sequence_eval, df_for_pred_eval,
                        config.INPUT_SEQ_LEN, config.OUTPUT_SEQ_LEN,
                        prediction_steps_eval, input_scaler, target_scaler, device
                    )

                    if predicted_states is not None and len(predicted_states) > 0:
                        actual_pred_steps = len(predicted_states)
                        true_states = true_states[:actual_pred_steps] # Ensure lengths match
                        model_mse = np.mean((predicted_states - true_states)**2)
                        print(f"多步预测整体 MSE ({actual_pred_steps} 步): {model_mse:.6f}")

                        # Generate physics comparison
                        print("进行物理模型模拟以用于比较...")
                        physics_x0 = df_eval_segment.iloc[config.INPUT_SEQ_LEN][['theta', 'theta_dot']].values
                        physics_time_eval = df_for_pred_eval['time'].iloc[:actual_pred_steps].values # Use actual steps
                        physics_predictions = None
                        if len(physics_time_eval) > 0:
                            physics_t_span = (physics_time_eval[0], physics_time_eval[-1])
                            physics_tau_values = df_for_pred_eval['tau'].iloc[:actual_pred_steps].values
                            _, physics_theta, physics_theta_dot = run_simulation(
                                pendulum_eval, physics_t_span, config.DT,
                                physics_x0, physics_tau_values, t_eval=physics_time_eval
                            )
                            if len(physics_theta) == len(physics_time_eval):
                                physics_predictions = np.stack([physics_theta, physics_theta_dot], axis=1)
                                physics_mse = np.mean((physics_predictions - true_states)**2)
                                print(f"物理模型 MSE ({actual_pred_steps} 步): {physics_mse:.6f}")

                        # Plot results
                        print("生成详细预测图表...")
                        plot_filename = f"multistep_eval_{config.MODEL_TYPE}"
                        plot_multi_step_prediction(
                            physics_time_eval, true_states, predicted_states,
                            physics_predictions, config.MODEL_TYPE,
                            config.FIGURES_DIR, filename_base=plot_filename
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

    eval_time = time.time() - eval_start_time
    print(f"模型评估完成，耗时: {eval_time:.2f}秒")

    # --- Final Summary ---
    total_time = time.time() - start_time
    print(f"\n{'='*20} 实验总结 {'='*20}")
    print(f"模型类型: {config.MODEL_TYPE}")
    print(f"特征工程 (SinCos): {config.USE_SINCOS_THETA}")
    print(f"数据生成耗时: {gen_time:.2f}s")
    print(f"数据预处理耗时: {data_prep_time:.2f}s")
    print(f"模型训练耗时: {train_time:.2f}s (最佳周期: {best_epoch})")
    print(f"模型评估耗时: {eval_time:.2f}s")
    print(f"总执行时间: {total_time:.2f}s")
    if np.isfinite(avg_val_loss_final): print(f"最终验证集 MSE (单步): {avg_val_loss_final:.6f}")
    if np.isfinite(model_mse): print(f"多步预测 MSE (评估段): {model_mse:.6f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    run_experiment()
