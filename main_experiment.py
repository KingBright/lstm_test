# main_experiment.py

import time
import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch.nn as nn

# Import configurations and utility functions
import config
import utils
from data_generation import PendulumSystem, generate_simulation_data, run_simulation
from data_preprocessing import prepare_data_for_training
from model import PureLSTM # Import the pure LSTM model
from training import train_model
from evaluation import evaluate_model, multi_step_prediction, plot_training_curves, plot_multi_step_prediction

def run_experiment():
    """
    Runs the complete workflow for training and evaluating the LSTM model.
    """
    start_time = time.time()

    # --- Setup ---
    utils.setup_logging_and_warnings()
    utils.setup_chinese_font()

    # Device Selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("检测到 MPS 加速")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("检测到 CUDA 加速")
    else:
        device = torch.device("cpu")
        print("使用 CPU")

    # Reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # --- Step 1: System Definition ---
    pendulum = PendulumSystem(m=config.PENDULUM_MASS, L=config.PENDULUM_LENGTH,
                              g=config.GRAVITY, c=config.DAMPING_COEFF)
    print("步骤1: 已创建单摆系统")

    # --- Step 2: Data Generation ---
    print("步骤2: 开始生成或加载扩展仿真数据...")
    gen_start_time = time.time()

    # Check for existing data file
    if os.path.exists(config.DATA_FILE_EXTENDED) and not config.FORCE_REGENERATE_DATA:
        print(f"加载已有的扩展仿真数据: {config.DATA_FILE_EXTENDED}...")
        df = pd.read_csv(config.DATA_FILE_EXTENDED)
    else:
        print(f"生成新的扩展仿真数据 (Force regenerate: {config.FORCE_REGENERATE_DATA})...")
        all_dfs = []
        for i, x0 in enumerate(config.INITIAL_CONDITIONS):
            print(f"  Simulating with initial condition {i+1}: {x0}...")
            df_single = generate_simulation_data(
                pendulum,
                t_span=config.T_SPAN,
                dt=config.DT,
                x0=x0,
                torque_type=config.TORQUE_TYPE
            )
            if not df_single.empty:
                all_dfs.append(df_single)
                print(f"  Simulation {i+1} done, {len(df_single)} points generated.")
            else:
                print(f"  Simulation {i+1} failed or produced no data.")

        if not all_dfs:
            print("Error: No simulation data could be generated. Exiting.")
            return
        df = pd.concat(all_dfs, ignore_index=True)
        print(f"总共生成 {len(df)} 数据点。")
        df.to_csv(config.DATA_FILE_EXTENDED, index=False)
        print(f"数据已保存到 {config.DATA_FILE_EXTENDED}")

    gen_time = time.time() - gen_start_time
    print(f"仿真数据处理完成，耗时: {gen_time:.2f}秒")

    # Plot sample data (optional)
    if not df.empty and config.PLOT_SAMPLE_DATA:
         utils.plot_simulation_data_sample(df, config.FIGURES_DIR)

    # --- Step 3: Data Preprocessing ---
    print("步骤3: 准备训练数据...")
    data_prep_start_time = time.time()
    data_tuple = prepare_data_for_training(
        df,
        sequence_length=config.SEQUENCE_LENGTH,
        test_split=config.TEST_SPLIT
    )
    (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
     X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,
     train_loader, test_loader, input_scaler, output_scaler) = data_tuple

    if X_train_tensor.nelement() == 0 or X_test_tensor.nelement() == 0:
        print("Error: Training or testing tensors are empty after data preparation. Exiting.")
        return

    data_prep_time = time.time() - data_prep_start_time
    print(f"训练数据准备完成，耗时: {data_prep_time:.2f}秒")
    print(f"训练集形状: X={X_train_scaled.shape}, y={y_train_scaled.shape}")
    print(f"测试集形状: X={X_test_scaled.shape}, y={y_test_scaled.shape}")

    # --- Step 4: Model Definition ---
    # Ensure input/output sizes are derived correctly
    if X_train_scaled.ndim != 3 or y_train_scaled.ndim != 2:
         print("Error: Unexpected dimensions for scaled training data.")
         return
    input_size = X_train_scaled.shape[2]
    output_size = y_train_scaled.shape[1]

    model = PureLSTM(
        input_size=input_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        output_size=output_size,
        dense_units=config.DENSE_UNITS,
        dropout=config.DROPOUT_RATE
    )
    print("步骤4: 已构建纯LSTM模型")
    print(model)

    # --- Step 5: Model Training ---
    print("步骤5: 开始训练模型...")
    train_start_time = time.time()
    train_losses, val_losses, best_epoch = train_model(
        model, train_loader, test_loader,
        num_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        device=device,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
        scheduler_factor=config.SCHEDULER_FACTOR,
        scheduler_patience=config.SCHEDULER_PATIENCE,
        weight_decay=config.WEIGHT_DECAY,
        model_save_path=config.MODEL_BEST_PATH, # Pass save path
        final_model_info_path=config.MODEL_FINAL_PATH # Pass final info save path
    )
    train_time = time.time() - train_start_time
    print(f"模型训练完成，耗时: {train_time:.2f}秒")

    # Plot training curves
    plot_training_curves(train_losses, val_losses, config.FIGURES_DIR)
    print("训练曲线已绘制并保存")

    if best_epoch == 0:
        print("Warning: Training did not yield improvement. Evaluation might use initial model state.")
        # Depending on `train_model`, the best model might not have been loaded back.
        # Consider loading the last state or exiting if no training occurred.

    # --- Step 6: Model Evaluation ---
    print("步骤6: 开始评估模型...")
    eval_start_time = time.time()

    # Ensure model is on the correct device for evaluation
    model.to(device)
    # Load the best model state if it exists (train_model might already do this)
    if os.path.exists(config.MODEL_BEST_PATH) and best_epoch > 0:
         try:
              model.load_state_dict(torch.load(config.MODEL_BEST_PATH, map_location=device))
              print(f"已加载最佳模型 (来自 Epoch {best_epoch})")
         except Exception as e:
              print(f"Warning: Failed to load best model state dict from {config.MODEL_BEST_PATH}. Error: {e}")
              print("Using model state from end of training.")
    elif best_epoch > 0:
         print(f"Warning: Best model path {config.MODEL_BEST_PATH} not found, but training indicated best epoch {best_epoch}. Using current model state.")
    else:
         print("评估使用的是训练结束时的模型状态 (无最佳验证模型)。")


    criterion = nn.MSELoss()
    avg_test_loss = evaluate_model(model, test_loader, criterion, device)

    # Multi-step Prediction Evaluation
    physics_predictions = None # Initialize
    predicted_states = np.array([])
    true_states = np.array([])

    if X_test_scaled.shape[0] > 0:
        start_idx_in_test = 0
        initial_sequence = X_test_scaled[start_idx_in_test]

        # Determine the starting index in the original dataframe 'df' for the test set
        num_train_sequences = len(X_train_scaled)
        start_index_in_df_for_pred = num_train_sequences + config.SEQUENCE_LENGTH

        if start_index_in_df_for_pred < len(df):
            df_for_pred = df.iloc[start_index_in_df_for_pred:].reset_index(drop=True)

            if not df_for_pred.empty:
                prediction_steps = min(len(X_test_scaled) - start_idx_in_test, len(df_for_pred))

                if prediction_steps > config.MIN_PREDICTION_STEPS: # Ensure minimum steps
                    print(f"执行多步预测，预测步数: {prediction_steps}")

                    predicted_states, true_states = multi_step_prediction(
                        model, initial_sequence, df_for_pred, config.SEQUENCE_LENGTH, prediction_steps,
                        input_scaler, output_scaler, device
                    )

                    # Generate high-fidelity physics comparison if prediction was successful
                    if len(predicted_states) > 0 and len(true_states) > 0:
                        print("生成高精度物理模型预测作为参考...")
                        physics_pendulum_ref = PendulumSystem(m=config.PENDULUM_MASS, L=config.PENDULUM_LENGTH,
                                                            g=config.GRAVITY, c=config.DAMPING_COEFF)
                        try:
                            # Use the start state corresponding to the first true_state
                            physics_x0 = df_for_pred.iloc[0][['theta', 'theta_dot']].values

                            # **Create the exact t_eval vector needed**
                            physics_time_eval = df_for_pred['time'].iloc[:prediction_steps].values
                            if len(physics_time_eval) == 0:
                                raise ValueError("Cannot create t_eval for physics comparison (zero length).")

                            # Define t_span based on t_eval extremes
                            physics_t_span = (physics_time_eval[0], physics_time_eval[-1])
                            # Get tau values corresponding ONLY to t_eval period
                            physics_tau_values = df_for_pred['tau'].iloc[:prediction_steps].values
                            # Determine dt from t_eval if possible, else use config default
                            physics_dt = physics_time_eval[1] - physics_time_eval[0] if len(physics_time_eval) > 1 else config.DT

                            # **Pass t_eval to run_simulation**
                            physics_time, physics_theta, physics_theta_dot = run_simulation(
                                physics_pendulum_ref, physics_t_span, physics_dt,
                                physics_x0, physics_tau_values, t_eval=physics_time_eval
                            )

                            # Check if simulation returned data matching t_eval length
                            if len(physics_time) == len(physics_time_eval):
                                physics_predictions = np.stack([physics_theta, physics_theta_dot], axis=1)
                            else:
                                print(f"Warning: Physics simulation returned {len(physics_time)} points, expected {len(physics_time_eval)}. Disabling comparison.")
                                physics_predictions = None


                            # Plot results including physics comparison (plot function handles None)
                            time_vector = physics_time_eval # Use the same time vector
                            plot_multi_step_prediction(
                                time_vector, true_states[:len(time_vector)], predicted_states[:len(time_vector)], # Ensure lengths match t_eval
                                physics_model_predictions=physics_predictions,
                                model_name="纯LSTM", save_dir=config.FIGURES_DIR
                            )
                            print(f"多步预测图表已保存到 {config.FIGURES_DIR}")

                        except ValueError as ve:
                            print(f"Error generating physics comparison: {ve}")
                            physics_predictions = None
                        except Exception as e:
                            print(f"Unexpected error during physics comparison generation: {e}")
                            physics_predictions = None
                            # Plot without physics if prediction succeeded but comparison failed
                            if len(predicted_states) > 0:
                                time_vector = df_for_pred['time'].iloc[:prediction_steps].values
                                plot_multi_step_prediction(time_vector, true_states, predicted_states, model_name="纯LSTM", save_dir=config.FIGURES_DIR)
                                print(f"多步预测图表 (无物理对比) 已保存到 {config.FIGURES_DIR}")

                    else:
                        print("Warning: Multi-step prediction returned no valid states. Skipping plotting.")
                else:
                    print(f"Warning: Insufficient steps ({prediction_steps}) for multi-step prediction based on available test data or dataframe slice.")
            else:
                print("Warning: DataFrame slice for prediction (df_for_pred) is empty. Skipping multi-step prediction.")
        else:
            print(f"Error: Calculated start index {start_index_in_df_for_pred} is out of bounds for the main DataFrame (length {len(df)}). Cannot perform multi-step prediction.")
    else:
        print("Warning: Test set is empty. Skipping multi-step prediction.")

    eval_time = time.time() - eval_start_time
    print(f"模型评估完成，耗时: {eval_time:.2f}秒")

    # --- Final Summary ---
    total_time = time.time() - start_time
    print(f"\n项目执行完毕！总耗时: {total_time:.2f}秒")
    print("\n===== 性能摘要 =====")
    print(f"数据生成时间: {gen_time:.2f}秒")
    print(f"数据准备时间: {data_prep_time:.2f}秒")
    print(f"模型训练时间: {train_time:.2f}秒")
    print(f"模型评估时间: {eval_time:.2f}秒")
    print(f"总执行时间: {total_time:.2f}秒")

    print("\n===== 模型性能摘要 =====")
    print(f"模型: PureLSTM")
    total_params_final = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数数量: {total_params_final:,}")
    print(f"序列长度: {config.SEQUENCE_LENGTH}")
    print(f"隐藏层大小: {config.HIDDEN_SIZE}")
    print(f"LSTM层数: {config.NUM_LAYERS}")
    if best_epoch > 0 and len(val_losses) >= best_epoch:
        best_val_loss_value = val_losses[best_epoch - 1]
        print(f"最佳验证损失 (MSE): {best_val_loss_value:.6f} (Epoch {best_epoch})")
    elif len(val_losses) > 0:
        print(f"训练期间验证损失未改善，最后损失: {val_losses[-1]:.6f}")
    print(f"测试集损失 (MSE): {avg_test_loss:.6f}")

    # Multi-step prediction summary
    if len(predicted_states) > 0 and len(true_states) == len(predicted_states):
        model_mse = np.mean((predicted_states - true_states) ** 2)
        print(f"\n模型多步预测 MSE: {model_mse:.6f}")
        if physics_predictions is not None and len(physics_predictions) == len(true_states):
            physics_mse = np.mean((physics_predictions - true_states) ** 2)
            print(f"纯物理模型 (RK45) 多步预测 MSE: {physics_mse:.6f}")
            if model_mse > 0 and physics_mse > 0:
                performance_ratio = physics_mse / model_mse
                print(f"性能比值 (物理 MSE / LSTM MSE): {performance_ratio:.2f}x")
                if performance_ratio > 1.0: print(f"LSTM 模型预测性能优于纯物理模型 {performance_ratio:.2f} 倍")
                else: print(f"LSTM 模型预测性能不如纯物理模型")
        else:
             print("无法计算或比较物理模型多步预测 MSE.")
    else:
        print("\n模型多步预测未成功执行或结果无效，无法计算 MSE.")

    print("\n项目完成！")

if __name__ == "__main__":
    run_experiment()