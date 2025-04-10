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
from data_generation import PendulumSystem, generate_simulation_data, run_simulation
# Import the NEW data prep function that takes separate train/val dfs
from data_preprocessing import prepare_train_val_data_from_dfs, create_sequences # Import create_sequences for eval part
from model import get_model # Use factory function
from training import train_model
# Import necessary evaluation functions
from evaluation import evaluate_model, multi_step_prediction, plot_training_curves, plot_multi_step_prediction

def run_experiment():
    """
    Runs the complete workflow with separate generation for Train and Validation sets.
    Includes plotting scenario comparisons.
    """
    start_time = time.time()

    # --- Setup ---
    utils.setup_logging_and_warnings()
    utils.setup_chinese_font()
    # Device Selection
    if torch.backends.mps.is_available():
        device = torch.device("mps"); print("检测到 MPS 加速")
    elif torch.cuda.is_available():
        device = torch.device("cuda"); print("检测到 CUDA 加速")
    else:
        device = torch.device("cpu"); print("使用 CPU")
    # Reproducibility
    torch.manual_seed(config.SEED); np.random.seed(config.SEED)

    # --- Step 1: System Definition ---
    pendulum = PendulumSystem(m=config.PENDULUM_MASS, L=config.PENDULUM_LENGTH,
                              g=config.GRAVITY, c=config.DAMPING_COEFF)
    print("步骤1: 已创建单摆系统")

    # --- Step 2: Generate Separate Train and Validation Data ---
    print("步骤2: 分别为训练集和验证集生成仿真数据...")
    gen_start_time = time.time()
    train_dfs = []
    val_dfs = []
    # Store data grouped by scenario for plotting later
    scenario_plot_data = defaultdict(lambda: {'train': [], 'val': []})
    total_train_points = 0
    total_val_points = 0

    # Check if data files exist and regeneration is not forced
    train_data_exists = os.path.exists(config.TRAIN_DATA_FILE)
    val_data_exists = os.path.exists(config.VAL_DATA_FILE)

    if train_data_exists and val_data_exists and not config.FORCE_REGENERATE_DATA:
        print(f"加载已有的训练数据: {config.TRAIN_DATA_FILE}")
        df_train = pd.read_csv(config.TRAIN_DATA_FILE)
        print(f"加载已有的验证数据: {config.VAL_DATA_FILE}")
        df_val = pd.read_csv(config.VAL_DATA_FILE)
        total_train_points = len(df_train)
        total_val_points = len(df_val)
        # If loading, scenario_plot_data will not be populated for plotting
        print("已加载预生成的数据文件。将跳过按场景绘图。")
        config.PLOT_SCENARIO_DATA = False # Disable scenario plotting if loading
    else:
        print(f"生成新的训练和验证数据 (Force regenerate: {config.FORCE_REGENERATE_DATA})...")
        print(f"场景: {config.SCENARIOS}")
        print(f"初始条件: {config.INITIAL_CONDITIONS_SPECIFIC}")
        print(f"训练时长: {config.T_SPAN_TRAIN}, 验证时长: {config.T_SPAN_VAL}")

        for scenario_type in config.SCENARIOS:
            print(f"  处理场景: {scenario_type}")
            scenario_train_dfs_for_plot = []
            scenario_val_dfs_for_plot = []
            for i, x0 in enumerate(config.INITIAL_CONDITIONS_SPECIFIC):
                # Generate training data for this combo
                df_train_single = generate_simulation_data(
                    pendulum, t_span=config.T_SPAN_TRAIN, dt=config.DT,
                    x0=x0, torque_type=scenario_type
                )
                if not df_train_single.empty:
                    train_dfs.append(df_train_single)
                    scenario_train_dfs_for_plot.append(df_train_single) # For plotting
                    total_train_points += len(df_train_single)
                else: print(f"    Warning: No train data generated for '{scenario_type}' with IC {x0}.")


                # Generate validation data for this combo
                df_val_single = generate_simulation_data(
                    pendulum, t_span=config.T_SPAN_VAL, dt=config.DT,
                    x0=x0, torque_type=scenario_type
                )
                if not df_val_single.empty:
                    val_dfs.append(df_val_single)
                    scenario_val_dfs_for_plot.append(df_val_single) # For plotting
                    total_val_points += len(df_val_single)
                else: print(f"    Warning: No val data generated for '{scenario_type}' with IC {x0}.")


            # Store for plotting this scenario
            scenario_plot_data[scenario_type]['train'] = scenario_train_dfs_for_plot
            scenario_plot_data[scenario_type]['val'] = scenario_val_dfs_for_plot

        if not train_dfs or not val_dfs: # Ensure both lists have data
            print("Error: 未能成功生成训练或验证数据。")
            return

        # Combine and save
        df_train = pd.concat(train_dfs, ignore_index=True)
        df_val = pd.concat(val_dfs, ignore_index=True)
        try:
            os.makedirs(config.MODELS_DIR, exist_ok=True) # Ensure dir exists for saving
            df_train.to_csv(config.TRAIN_DATA_FILE, index=False)
            df_val.to_csv(config.VAL_DATA_FILE, index=False)
            print(f"训练数据已保存到 {config.TRAIN_DATA_FILE} ({len(df_train)} points)")
            print(f"验证数据已保存到 {config.VAL_DATA_FILE} ({len(df_val)} points)")
        except Exception as e:
            print(f"保存数据文件时出错: {e}")
            # Decide if error is critical
            # return

    gen_time = time.time() - gen_start_time
    print(f"数据生成/加载完成，耗时: {gen_time:.2f}秒")
    print(f"总训练数据点: {total_train_points}, 总验证数据点: {total_val_points}")

    # --- Plotting Generated Data by Scenario (if generated) ---
    if config.PLOT_SCENARIO_DATA and scenario_plot_data:
        print("\n--- 绘制每个场景的训练/验证数据对比图 (仅角度) ---")
        for scenario_name, plot_data in scenario_plot_data.items():
            # Call the new plotting function from utils
            utils.plot_scenario_comparison(
                scenario_name,
                plot_data['train'], # List of DFs for train
                plot_data['val'],   # List of DFs for val
                config.FIGURES_DIR
            )
    elif config.PLOT_SCENARIO_DATA:
        print("跳过按场景绘图，因为数据是加载的或生成失败。")


    # --- Step 3: Prepare DataLoaders from df_train and df_val ---
    print("\n步骤3: 从训练和验证 DataFrame 创建 DataLoaders...")
    data_prep_start_time = time.time()
    # Call the NEW data prep function that takes df_train, df_val
    data_loaders_tuple = prepare_train_val_data_from_dfs(
        df_train, df_val, # Pass the two dataframes
        sequence_length=config.SEQUENCE_LENGTH,
        seed=config.SEED
        # Scaler types/paths are handled inside using config
    )

    # Check if data prep was successful
    if data_loaders_tuple is None or data_loaders_tuple[0] is None:
        print("Error: Failed to create datasets and loaders. Exiting.")
        return

    train_loader, val_loader, input_scaler, target_scaler = data_loaders_tuple
    data_prep_time = time.time() - data_prep_start_time
    print(f"数据准备完成，耗时: {data_prep_time:.2f}秒")
    train_loader_len_check = len(train_loader) if train_loader else 0
    val_loader_len_check = len(val_loader) if val_loader else 0
    print(f"Train loader batches: {train_loader_len_check}, Val loader batches: {val_loader_len_check}")
    if train_loader_len_check == 0: print("Error: Training loader is empty."); return

    # --- Step 4: Model Definition ---
    # Determine input/output size from scalers
    try:
        input_size = input_scaler.n_features_in_
        output_size = target_scaler.n_features_in_ # Target scaler determines output size
    except AttributeError: print("Error: Scalers seem invalid."); return
    try:
        # Use the factory function to get the model based on config
        model = get_model(model_type=config.MODEL_TYPE,
                          input_size=input_size,
                          output_size=output_size)
        print("Model created successfully."); print(model)
    except Exception as e: print(f"Error creating model: {e}"); return

    # --- Step 5: Model Training ---
    print("\n步骤5: 开始训练模型...")
    train_start_time = time.time()
    # Pass val_loader to train_model
    train_losses, val_losses, best_epoch = train_model(
        model, train_loader, val_loader, # Pass val_loader here
        device=device,
        model_save_path=config.MODEL_BEST_PATH,
        final_model_info_path=config.MODEL_FINAL_PATH
        # Other params use defaults from config via train_model signature
    )
    train_time = time.time() - train_start_time
    print(f"模型训练完成，耗时: {train_time:.2f}秒")

    # Plot training curves using the function from evaluation module
    plot_training_curves(train_losses, val_losses, config.FIGURES_DIR)
    print("训练曲线已绘制并保存")
    if best_epoch == 0: print("Warning: Training did not yield improvement based on validation loss.")

    # --- Step 6: Model Evaluation (on Validation Set & a Test Segment) ---
    print("\n步骤6: 开始在验证集上评估模型...")
    eval_start_time = time.time()

    # Load best model state if training was successful
    model.to(device) # Ensure model is on device
    if os.path.exists(config.MODEL_BEST_PATH) and best_epoch > 0:
        try: model.load_state_dict(torch.load(config.MODEL_BEST_PATH, map_location=device)); print(f"已加载最佳模型 (来自 Epoch {best_epoch})")
        except Exception as e: print(f"Warning: Failed to load best model state. Error: {e}")
    else: print("使用训练结束时的模型状态进行评估。")

    # Evaluate single-step MSE on the validation set
    criterion = nn.MSELoss()
    avg_val_loss_final = evaluate_model(model, val_loader, criterion, device) if val_loader else float('nan')
    if np.isfinite(avg_val_loss_final): print(f"最终模型在验证集上的单步 MSE: {avg_val_loss_final:.6f}")
    else: print("无法计算验证集单步 MSE。")

    # --- Multi-Step Prediction Evaluation (using a specific validation segment) ---
    # This section uses the START of the validation data file for a consistent multi-step test
    print("\n运行多步预测（使用验证数据文件的开头部分）...")
    physics_predictions = None
    predicted_states = np.array([])
    true_states = np.array([])
    try:
        # Load the validation data file generated during training
        if os.path.exists(config.VAL_DATA_FILE):
             df_val_eval = pd.read_csv(config.VAL_DATA_FILE)
             # Use the beginning of the validation file for evaluation
             # Determine how many steps based on available data vs limit
             max_eval_steps = len(df_val_eval) - config.SEQUENCE_LENGTH
             if max_eval_steps >= config.MIN_PREDICTION_STEPS:
                 prediction_steps_eval = max_eval_steps # Predict full available length
                 # Optional: Limit steps further if needed for speed
                 # prediction_steps_eval = min(max_eval_steps, 1000)

                 eval_data_values = df_val_eval[['theta', 'theta_dot', 'tau']].values
                 # Create sequences from this validation data segment (absolute states for X)
                 X_eval, _ = create_sequences(eval_data_values, config.SEQUENCE_LENGTH, predict_delta=False)

                 if len(X_eval) > 0:
                     X_eval_reshaped = X_eval.reshape(-1, X_eval.shape[2])
                     X_eval_scaled_reshaped = input_scaler.transform(X_eval_reshaped) # Use fitted scaler
                     X_eval_scaled = X_eval_scaled_reshaped.reshape(X_eval.shape)

                     initial_sequence_eval = X_eval_scaled[0] # Start from first sequence of val set
                     # df_for_pred starts after the initial sequence used
                     df_for_pred_eval = df_val_eval.iloc[config.SEQUENCE_LENGTH:].reset_index(drop=True)
                     # Adjust prediction steps based on df_for_pred length
                     prediction_steps_eval = min(prediction_steps_eval, len(df_for_pred_eval))

                     if prediction_steps_eval >= config.MIN_PREDICTION_STEPS:
                          predicted_states, true_states = multi_step_prediction(
                              model, initial_sequence_eval, df_for_pred_eval, config.SEQUENCE_LENGTH, prediction_steps_eval,
                              input_scaler, target_scaler, device # Pass correct target scaler
                          )

                          # Generate physics comparison for this segment
                          if len(predicted_states) > 0 and len(true_states) == len(predicted_states):
                               physics_pendulum_ref = PendulumSystem(m=config.PENDULUM_MASS, L=config.PENDULUM_LENGTH, g=config.GRAVITY, c=config.DAMPING_COEFF)
                               physics_x0 = df_val_eval.iloc[config.SEQUENCE_LENGTH][['theta', 'theta_dot']].values # State at start of prediction
                               physics_time_eval = df_for_pred_eval['time'].iloc[:prediction_steps_eval].values # Match steps
                               physics_t_span = (physics_time_eval[0], physics_time_eval[-1])
                               physics_tau_values = df_for_pred_eval['tau'].iloc[:prediction_steps_eval].values
                               physics_dt = config.DT
                               physics_time, physics_theta, physics_theta_dot = run_simulation(physics_pendulum_ref, physics_t_span, physics_dt, physics_x0, physics_tau_values, t_eval=physics_time_eval)
                               if len(physics_time) == len(physics_time_eval): physics_predictions = np.stack([physics_theta, physics_theta_dot], axis=1)
                               else: physics_predictions = None
                               # Plot
                               plot_multi_step_prediction(physics_time_eval, true_states, predicted_states, physics_predictions, config.MODEL_TYPE, config.FIGURES_DIR)
                               print(f"多步预测图表 (验证集起始段) 已保存到 {config.FIGURES_DIR}")
                     else: print("多步预测未能生成有效结果。")
                 else: print("无法为评估段创建序列。")
             else: print(f"验证数据文件中的可用步数 ({max_eval_steps}) 少于最小要求 ({config.MIN_PREDICTION_STEPS})，跳过多步预测。")
        else:
             print(f"验证数据文件 {config.VAL_DATA_FILE} 未找到，跳过多步预测评估。")

    except Exception as eval_msp_e:
         print(f"执行多步预测评估时出错: {eval_msp_e}")
         import traceback
         traceback.print_exc()

    eval_time = time.time() - eval_start_time
    print(f"模型评估完成，耗时: {eval_time:.2f}秒")

    # --- Final Summary ---
    total_time = time.time() - start_time
    print(f"\n项目执行完毕！总耗时: {total_time:.2f}秒")
    print("\n===== 性能摘要 =====")
    print(f"数据生成/加载时间: {gen_time:.2f}秒") # Adjusted label
    print(f"数据准备时间: {data_prep_time:.2f}秒")
    print(f"模型训练时间: {train_time:.2f}秒")
    print(f"模型评估时间: {eval_time:.2f}秒")
    print(f"总执行时间: {total_time:.2f}秒")

    print("\n===== 模型性能摘要 =====")
    current_params = config.get_current_model_params()
    print(f"模型类型: {config.MODEL_TYPE}")
    total_params_final = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数数量: {total_params_final:,}")
    print(f"序列长度: {config.SEQUENCE_LENGTH}")
    print(f"隐藏层大小: {current_params.get('hidden_size', 'N/A')}")
    print(f"RNN层数: {current_params.get('num_layers', 'N/A')}")
    if best_epoch > 0 and len(val_losses) >= best_epoch and np.isfinite(val_losses[best_epoch - 1]):
        best_val_loss_value = val_losses[best_epoch - 1]
        print(f"最佳验证损失 (MSE): {best_val_loss_value:.6f} (Epoch {best_epoch})")
    elif len(val_losses) > 0 and np.isfinite(val_losses[-1]):
        print(f"训练期间验证损失未改善或无效，最后验证损失: {val_losses[-1]:.6f}")
    print(f"验证集单步 MSE (最终模型): {avg_val_loss_final:.6f}")

    # Multi-step prediction summary
    if len(predicted_states) > 0 and len(true_states) == len(predicted_states):
        model_mse = np.mean((predicted_states - true_states) ** 2)
        print(f"\n模型多步预测 MSE (验证集起始段): {model_mse:.6f}") # Clarify segment
        if physics_predictions is not None and len(physics_predictions) == len(true_states):
            physics_mse = np.mean((physics_predictions - true_states) ** 2)
            print(f"纯物理模型 (RK45) 多步预测 MSE (验证集起始段): {physics_mse:.6f}")
            if model_mse > 0 and physics_mse > 0:
                performance_ratio = physics_mse / model_mse
                print(f"性能比值 (物理 MSE / 模型 MSE): {performance_ratio:.2f}x")
                if performance_ratio > 1.0: print(f"模型预测性能优于纯物理模型 {performance_ratio:.2f} 倍 (在该评估段上)")
                else: print(f"模型预测性能不如纯物理模型 (在该评估段上)")
        else: print("无法计算或比较物理模型多步预测 MSE.")
    else: print("\n模型多步预测 (评估段) 未成功执行或结果无效，无法计算 MSE.")

    print("\n项目完成！")

if __name__ == "__main__":
    run_experiment()
