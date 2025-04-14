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
from data_generation import PendulumSystem, run_simulation, generate_simulation_data
# 使用改进的数据生成方法
from improved_data_generation import generate_improved_dataset, analyze_dataset_coverage
from data_preprocessing import prepare_timesplit_seq2seq_data, create_sequences
from model import get_model # Use factory function
from training import train_model
# Import necessary evaluation functions
from evaluation import evaluate_model, multi_step_prediction, plot_training_curves, plot_multi_step_prediction

# Helper function for the multi-step evaluation block to reduce duplication
def _run_multi_step_eval(model_to_eval, model_label, device, input_scaler, target_scaler):
    """Runs the multi-step prediction evaluation for a given model state."""
    print(f"\n--- Running Multi-Step Evaluation for: {model_label} ---")
    predicted_states, true_states, physics_predictions = None, None, None
    model_mse = np.nan
    try:
        print("Generating random segment for evaluation...")
        pendulum_eval = PendulumSystem()
        eval_x0 = [np.random.uniform(*config.THETA_RANGE), np.random.uniform(*config.THETA_DOT_RANGE)]
        eval_duration = config.SIMULATION_DURATION_MEDIUM # Use medium duration for consistency
        df_eval_segment = generate_simulation_data(
            pendulum_eval, t_span=(0, eval_duration), dt=config.DT,
            x0=eval_x0, torque_type=config.TORQUE_TYPE
        )

        min_required_len = config.INPUT_SEQ_LEN + config.MIN_PREDICTION_STEPS
        if not df_eval_segment.empty and len(df_eval_segment) >= min_required_len:
            print(f"Generated evaluation segment: {len(df_eval_segment)} points")
            eval_data_values = df_eval_segment[['theta', 'theta_dot', 'tau']].values
            use_sincos_eval = config.USE_SINCOS_THETA

            X_eval, _ = create_sequences(
                eval_data_values, simulation_boundaries=None,
                input_seq_len=config.INPUT_SEQ_LEN, output_seq_len=config.OUTPUT_SEQ_LEN,
                use_sincos=use_sincos_eval
            )

            if len(X_eval) > 0:
                if input_scaler.n_features_in_ != X_eval.shape[2]:
                    raise ValueError(f"Eval data feature mismatch: Scaler={input_scaler.n_features_in_}, Data={X_eval.shape[2]}")

                X_eval_scaled = input_scaler.transform(X_eval.reshape(-1, X_eval.shape[2])).reshape(X_eval.shape)
                initial_sequence_eval = X_eval_scaled[0]
                df_for_pred_eval = df_eval_segment.iloc[config.INPUT_SEQ_LEN:].reset_index(drop=True)
                prediction_steps_eval = min(len(df_for_pred_eval), 1000) # Limit steps

                if prediction_steps_eval >= config.MIN_PREDICTION_STEPS:
                    print(f"Executing {prediction_steps_eval} prediction steps...")
                    # Ensure model is in eval mode and on correct device
                    model_to_eval.eval()
                    model_to_eval.to(device)
                    predicted_states, true_states = multi_step_prediction(
                        model_to_eval, initial_sequence_eval, df_for_pred_eval,
                        config.INPUT_SEQ_LEN, config.OUTPUT_SEQ_LEN,
                        prediction_steps_eval, input_scaler, target_scaler, device
                    )

                    if predicted_states is not None and len(predicted_states) > 0:
                        actual_pred_steps = len(predicted_states)
                        true_states = true_states[:actual_pred_steps]
                        model_mse = np.mean((predicted_states - true_states)**2)
                        print(f"  MSE ({model_label}, {actual_pred_steps} steps): {model_mse:.6f}")

                        # Physics comparison
                        print("Generating physics comparison...")
                        physics_x0 = df_eval_segment.iloc[config.INPUT_SEQ_LEN][['theta', 'theta_dot']].values
                        physics_time_eval = df_for_pred_eval['time'].iloc[:actual_pred_steps].values
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
                                print(f"  Physics Model MSE ({actual_pred_steps} steps): {physics_mse:.6f}")

                        # Plotting
                        print("Generating prediction plot...")
                        plot_filename = f"multistep_eval_{config.MODEL_TYPE}_{model_label.replace(' ','_')}"
                        plot_multi_step_prediction(
                            physics_time_eval, true_states, predicted_states,
                            physics_predictions, f"{config.MODEL_TYPE} ({model_label})",
                            config.FIGURES_DIR, filename_base=plot_filename
                        )
                        print(f"  Plot saved: {config.FIGURES_DIR}/{plot_filename}.png")
                    else: print("Multi-step prediction failed.")
                else: print(f"Insufficient steps for multi-step eval ({prediction_steps_eval} < {config.MIN_PREDICTION_STEPS}).")
            else: print("Could not create sequences for evaluation segment.")
        else: print(f"Generated evaluation segment too short ({len(df_eval_segment)} points).")
    except Exception as eval_err:
        print(f"Error during multi-step evaluation for {model_label}: {eval_err}")
        import traceback
        traceback.print_exc()
    # Return the calculated MSE for summary
    return model_mse


def run_experiment():
    """
    Runs the complete workflow including training and evaluation of final/best models.
    """
    start_time = time.time()
    utils.setup_logging_and_warnings()
    utils.setup_chinese_font()

    # --- Setup Device ---
    if torch.backends.mps.is_available(): device = torch.device("mps"); os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'; print(f"使用 MPS 加速 (M2 Max)")
    elif torch.cuda.is_available(): device = torch.device("cuda"); print(f"使用 CUDA 加速")
    else: device = torch.device("cpu"); print(f"使用 CPU 计算")
    torch.manual_seed(config.SEED); np.random.seed(config.SEED)

    # --- Step 1: System Definition ---
    pendulum = PendulumSystem(m=config.PENDULUM_MASS, L=config.PENDULUM_LENGTH, g=config.GRAVITY, c=config.DAMPING_COEFF)
    print("步骤1: 已创建单摆系统")

    # --- Step 2: Generate/Load Data ---
    print("步骤2: 生成/加载训练数据...")
    gen_start_time = time.time()
    df_all, simulation_boundaries = None, []
    if os.path.exists(config.COMBINED_DATA_FILE) and not config.FORCE_REGENERATE_DATA:
        print(f"加载数据: {config.COMBINED_DATA_FILE}...")
        try:
            df_all = pd.read_csv(config.COMBINED_DATA_FILE)
            boundaries_file = config.COMBINED_DATA_FILE.replace('.csv', '_boundaries.npy')
            if os.path.exists(boundaries_file): simulation_boundaries = list(np.load(boundaries_file))
            else: simulation_boundaries = None; print("警告: 未找到边界文件。")
            print(f"加载完成: {len(df_all)} 点, {len(simulation_boundaries) if simulation_boundaries else 0} 边界。")
        except Exception as e:
            print(f"加载失败: {e}. 将重新生成数据。"); df_all = None; simulation_boundaries = []; config.FORCE_REGENERATE_DATA = True
    else: config.FORCE_REGENERATE_DATA = True

    if df_all is None or config.FORCE_REGENERATE_DATA:
        print("生成新数据...")
        df_all, simulation_boundaries = generate_improved_dataset(
            target_sequences=config.TARGET_SEQUENCES, dt=config.DT, output_file=config.COMBINED_DATA_FILE
        )
        if df_all is not None and not df_all.empty:
            analyze_dataset_coverage(df_all, config.FIGURES_DIR)
        else: print("错误: 数据生成失败!"); return
    gen_time = time.time() - gen_start_time
    print(f"数据生成/加载完成，耗时: {gen_time:.2f}s")
    if df_all is None or df_all.empty: print("数据无效，无法继续。"); return

    # --- Step 3: Prepare DataLoaders ---
    print("\n步骤3: 准备 DataLoaders...")
    data_prep_start_time = time.time()
    data_loaders_tuple = prepare_timesplit_seq2seq_data(
        df_all, simulation_boundaries=simulation_boundaries,
        input_sequence_length=config.INPUT_SEQ_LEN, output_sequence_length=config.OUTPUT_SEQ_LEN,
        val_split_ratio=config.VALIDATION_SPLIT, seed=config.SEED
    )
    if data_loaders_tuple is None or data_loaders_tuple[0] is None: print("错误: 创建 DataLoaders 失败。"); return
    train_loader, val_loader, input_scaler, target_scaler = data_loaders_tuple
    data_prep_time = time.time() - data_prep_start_time
    print(f"数据准备完成，耗时: {data_prep_time:.2f}s")
    if not train_loader: print("错误: 训练 DataLoader 为空。"); return

    # --- Step 4: Model Definition ---
    print("\n步骤4: 创建模型...")
    try:
        input_size = input_scaler.n_features_in_ if hasattr(input_scaler, 'n_features_in_') else (4 if config.USE_SINCOS_THETA else 3)
        output_size = target_scaler.n_features_in_ if hasattr(target_scaler, 'n_features_in_') else 2
        print(f"  确认: input_size={input_size}, output_size={output_size}")
        model = get_model(model_type=config.MODEL_TYPE, input_size=input_size, output_size=output_size)
        print(f"模型创建成功! 类型: {config.MODEL_TYPE}")
    except Exception as e: print(f"错误: 创建模型失败: {e}"); traceback.print_exc(); return

    # --- Step 5: Model Training ---
    print("\n步骤5: 开始训练模型...")
    train_start_time = time.time()
    # Note: Early stopping is now disabled via config, will run for full NUM_EPOCHS
    train_losses, val_losses, best_epoch = train_model(
        model, train_loader, val_loader, device=device,
        model_save_path=config.MODEL_BEST_PATH,
        final_model_info_path=config.MODEL_FINAL_PATH
    )
    train_time = time.time() - train_start_time
    print(f"模型训练完成，耗时: {train_time:.2f}s")
    plot_training_curves(train_losses, val_losses, config.FIGURES_DIR)
    print("训练曲线已绘制并保存")
    # best_epoch might be 0 if validation loss never improved or val_loader was None

    # --- Step 6: Model Evaluation (Final and Best Model) ---
    print("\n步骤6: 开始评估模型...")
    eval_start_time = time.time()
    final_model_mse = np.nan
    best_model_mse = np.nan

    # --- Evaluate Final Model State ---
    # The 'model' object currently holds the state from the last epoch
    print("\nEvaluating Final Model State (End of Training)...")
    final_model_mse = _run_multi_step_eval(model, "Final", device, input_scaler, target_scaler)

    # --- Evaluate Best Model State (if saved) ---
    if os.path.exists(config.MODEL_BEST_PATH) and best_epoch > 0:
        print(f"\nLoading and Evaluating Best Model State (from Epoch {best_epoch})...")
        try:
            # Create a new model instance or load state into the existing one
            # Loading into existing 'model' object is fine here
            if device.type == 'mps':
                state_dict = torch.load(config.MODEL_BEST_PATH, map_location='mps', weights_only=True)
                model.load_state_dict(state_dict)
                torch.mps.synchronize()
            else:
                state_dict = torch.load(config.MODEL_BEST_PATH, map_location=device, weights_only=True)
                model.load_state_dict(state_dict)
            print("最佳模型状态加载成功。")
            # Run evaluation on the loaded best model state
            best_model_mse = _run_multi_step_eval(model, f"Best_Epoch_{best_epoch}", device, input_scaler, target_scaler)
        except Exception as e:
            print(f"加载或评估最佳模型时出错: {e}")
            traceback.print_exc()
    elif best_epoch == 0:
         print("\n未找到或未保存最佳模型 (可能验证损失从未改善或无验证集)。跳过最佳模型评估。")
    else:
        print(f"\n找不到最佳模型文件: {config.MODEL_BEST_PATH}。跳过最佳模型评估。")


    eval_time = time.time() - eval_start_time
    print(f"\n模型评估完成，耗时: {eval_time:.2f}秒")

    # --- Final Summary ---
    total_time = time.time() - start_time
    print(f"\n{'='*20} 实验总结 {'='*20}")
    print(f"模型类型: {config.MODEL_TYPE} (h{model.hidden_size}, l{model.num_layers})") # Assuming model has these attributes
    print(f"特征工程 (SinCos): {config.USE_SINCOS_THETA}")
    print(f"数据生成/加载耗时: {gen_time:.2f}s")
    print(f"数据预处理耗时: {data_prep_time:.2f}s")
    print(f"模型训练耗时: {train_time:.2f}s (共 {config.NUM_EPOCHS} 周期, 最佳周期: {best_epoch if best_epoch > 0 else 'N/A'})")
    print(f"模型评估耗时: {eval_time:.2f}s")
    print(f"总执行时间: {total_time:.2f}s")
    # Report both MSEs if available
    if not np.isnan(final_model_mse): print(f"多步预测 MSE (最终模型): {final_model_mse:.6f}")
    if not np.isnan(best_model_mse): print(f"多步预测 MSE (最佳模型): {best_model_mse:.6f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    run_experiment()
