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
import utils # Need utils for plotting
# Import functions/classes from respective modules
from data_generation import PendulumSystem, generate_simulation_data, run_simulation
# Import the NEW shuffle-then-split data prep function and create_sequences
from data_preprocessing import prepare_shuffled_train_val_data, create_sequences
from model import get_model # Use factory function
from training import train_model
# Import necessary evaluation functions
from evaluation import evaluate_model, multi_step_prediction, plot_training_curves, plot_multi_step_prediction

def run_experiment():
    """
    Runs the complete workflow using highly randomized data generation,
    shuffle-then-split preprocessing (operating on list of DFs),
    Seq2Seq model training/evaluation.
    Physics validation removed.
    """
    start_time = time.time()

    # --- Setup ---
    utils.setup_logging_and_warnings()
    utils.setup_chinese_font()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(config.SEED); np.random.seed(config.SEED)

    # --- Step 1: System Definition ---
    pendulum = PendulumSystem(m=config.PENDULUM_MASS, L=config.PENDULUM_LENGTH,
                              g=config.GRAVITY, c=config.DAMPING_COEFF)
    print("步骤1: 已创建单摆系统")

    # --- Step 2: Generate Highly Randomized Data via Short Simulations ---
    print("步骤2: 通过大量短时仿真生成高度随机数据...")
    gen_start_time = time.time()
    all_dfs = [] # <<<--- Store individual DataFrames in this list
    total_points = 0
    df_combined_for_saving = None # Initialize

    # Check if combined file exists and regeneration is not forced (for potential loading in future)
    if os.path.exists(config.COMBINED_DATA_FILE) and not config.FORCE_REGENERATE_DATA:
        print(f"加载已有的合并数据文件: {config.COMBINED_DATA_FILE}...")
        try:
            # Load the combined file first to get df_all for potential use/saving check
            df_all_loaded = pd.read_csv(config.COMBINED_DATA_FILE)
            total_points = len(df_all_loaded)
            print(f"已加载 {total_points} 数据点。")
            # IMPORTANT: We still need the list of DFs for prepare_shuffled_train_val_data
            # If we load the combined file, we cannot easily reconstruct the list.
            # Therefore, we will regenerate the list even if the file exists,
            # unless a more complex loading/saving mechanism for the list is implemented.
            print("警告: 找到合并文件，但当前流程需要重新生成 DataFrame 列表以进行预处理。将重新生成...")
            config.FORCE_REGENERATE_DATA = True # Force regeneration to get the list
        except Exception as e:
            print(f"加载合并数据文件时出错: {e}. 将尝试重新生成。")
            config.FORCE_REGENERATE_DATA = True

    # Generate new data if needed
    if not all_dfs: # Check if list is empty (needs generation)
        print(f"生成新的随机数据 (Force regenerate: {config.FORCE_REGENERATE_DATA})...")
        print(f"将执行 {config.NUM_SIMULATIONS} 次短时仿真...")
        for i in range(config.NUM_SIMULATIONS):
            x0 = [np.random.uniform(*config.THETA_RANGE), np.random.uniform(*config.THETA_DOT_RANGE)]
            df_single = generate_simulation_data(pendulum, t_span=config.T_SPAN_SHORT, dt=config.DT, x0=x0, torque_type=config.TORQUE_TYPE)
            # --- Physics validation REMOVED ---
            if not df_single.empty:
                all_dfs.append(df_single) # <<<--- Append individual df to the list
                total_points += len(df_single)
            if (i + 1) % 500 == 0: print(f"  已完成 {i+1}/{config.NUM_SIMULATIONS} 次仿真...")

        if not all_dfs: print("Error: 未能成功生成任何仿真数据。"); return

        print(f"所有仿真数据片段已生成，总数据点: {total_points}, 总片段数: {len(all_dfs)}")

        # --- Optionally save the combined data (for evaluate_saved_model.py or inspection) ---
        try:
            df_combined_for_saving = pd.concat(all_dfs, ignore_index=True)
            os.makedirs(config.MODELS_DIR, exist_ok=True)
            df_combined_for_saving.to_csv(config.COMBINED_DATA_FILE, index=False)
            print(f"合并后的数据已保存到 {config.COMBINED_DATA_FILE} (供参考或独立评估使用)")
        except Exception as e: print(f"保存合并数据文件时出错: {e}")

    gen_time = time.time() - gen_start_time
    print(f"数据生成完成，耗时: {gen_time:.2f}秒")
    if not all_dfs: print("数据生成失败，无法继续。"); return # Check if list is populated

    # --- Physics Validation Step REMOVED ---

    # --- Step 3: Prepare DataLoaders using Shuffle-then-Split ---
    print("\n步骤3: 创建序列，打乱并分割训练/验证集...")
    data_prep_start_time = time.time()
    # --- VVVVVV 调用新的数据准备函数，并传递 DataFrame 列表 VVVVVV ---
    data_loaders_tuple = prepare_shuffled_train_val_data(
        all_dfs, # <<<--- 传递 DataFrame 列表 all_dfs
        sequence_length=config.INPUT_SEQ_LEN,
        output_sequence_length=config.OUTPUT_SEQ_LEN,
        val_fraction=config.VAL_SET_FRACTION,
        seed=config.SEED
    )
    # --- ^^^^^^ 调用新的数据准备函数，并传递 DataFrame 列表 ^^^^^^ ---

    if data_loaders_tuple is None or data_loaders_tuple[0] is None: print("Error: Failed to create datasets and loaders."); return
    train_loader, val_loader, input_scaler, target_scaler = data_loaders_tuple
    data_prep_time = time.time() - data_prep_start_time
    print(f"数据准备完成，耗时: {data_prep_time:.2f}秒")
    train_loader_len_check = len(train_loader) if train_loader else 0; val_loader_len_check = len(val_loader) if val_loader else 0
    print(f"Train loader batches: {train_loader_len_check}, Val loader batches: {val_loader_len_check}")
    if train_loader_len_check == 0: print("Error: Training loader is empty."); return

    # --- Step 4: Model Definition ---
    # ... (Model definition logic remains the same) ...
    try:
        input_size = 4 if config.USE_SINCOS_THETA else 3; output_size = 2;
        if hasattr(input_scaler, 'n_features_in_') and input_scaler.n_features_in_ != input_size: input_size = input_scaler.n_features_in_; print(f"  使用来自 scaler 的 input_size = {input_size}")
        if hasattr(target_scaler, 'n_features_in_') and target_scaler.n_features_in_ != output_size: output_size = target_scaler.n_features_in_; print(f"  使用来自 scaler 的 output_size = {output_size}")
    except AttributeError: print("Error: Scalers invalid."); return
    try: model = get_model(model_type=config.MODEL_TYPE, input_size=input_size, output_size=output_size); print("Model created successfully."); print(model)
    except Exception as e: print(f"Error creating model: {e}"); return


    # --- Step 5: Model Training ---
    # ... (Model training call remains the same) ...
    print("\n步骤5: 开始训练模型...")
    train_start_time = time.time(); train_losses, val_losses, best_epoch = train_model(model, train_loader, val_loader, device=device, model_save_path=config.MODEL_BEST_PATH, final_model_info_path=config.MODEL_FINAL_PATH); train_time = time.time() - train_start_time
    print(f"模型训练完成，耗时: {train_time:.2f}秒"); plot_training_curves(train_losses, val_losses, config.FIGURES_DIR); print("训练曲线已绘制并保存")
    if best_epoch == 0: print("Warning: Training did not yield improvement.")

    # --- Step 6: Model Evaluation (Simplified for Seq2Seq) ---
    # ... (Evaluation logic remains the same, using a regenerated segment) ...
    print("\n步骤6: 开始在验证集上评估模型...")
    eval_start_time = time.time(); model.to(device)
    if os.path.exists(config.MODEL_BEST_PATH) and best_epoch > 0:
        try: model.load_state_dict(torch.load(config.MODEL_BEST_PATH, map_location=device, weights_only=True)); print(f"已加载最佳模型 (来自 Epoch {best_epoch})") # Added weights_only
        except Exception as e: print(f"Warning: Failed to load best model state. Error: {e}")
    else: print("使用训练结束时的模型状态进行评估。")
    criterion = nn.MSELoss(); avg_val_loss_final = evaluate_model(model, val_loader, criterion, device) if val_loader else float('nan')
    if np.isfinite(avg_val_loss_final): print(f"最终模型在验证集上的单步 MSE: {avg_val_loss_final:.6f}")
    else: print("无法计算验证集单步 MSE。")
    print("\n运行多步预测（使用新生成的随机片段进行检查）...")
    predicted_states, true_states, physics_predictions = None, None, None; model_mse = np.nan
    try:
        pendulum_eval = PendulumSystem(); eval_x0 = [np.random.uniform(*config.THETA_RANGE), np.random.uniform(*config.THETA_DOT_RANGE)]; eval_duration = 20
        df_eval_segment = generate_simulation_data(pendulum_eval, t_span=(0, eval_duration), dt=config.DT, x0=eval_x0, torque_type=config.TORQUE_TYPE)
        if not df_eval_segment.empty and len(df_eval_segment) > config.INPUT_SEQ_LEN + config.OUTPUT_SEQ_LEN:
            eval_data_values = df_eval_segment[['theta', 'theta_dot', 'tau']].values
            use_sincos_eval = config.USE_SINCOS_THETA
            X_eval, _ = create_sequences(eval_data_values, config.INPUT_SEQ_LEN, config.OUTPUT_SEQ_LEN, use_sincos=use_sincos_eval) # Removed predict_delta
            if len(X_eval) > 0:
                if input_scaler.n_features_in_ != X_eval.shape[2]: raise ValueError(f"Input scaler/data feature mismatch: {input_scaler.n_features_in_} vs {X_eval.shape[2]}")
                X_eval_scaled = input_scaler.transform(X_eval.reshape(-1, X_eval.shape[2])).reshape(X_eval.shape)
                initial_sequence_eval = X_eval_scaled[0]; df_for_pred_eval = df_eval_segment.iloc[config.INPUT_SEQ_LEN:].reset_index(drop=True)
                prediction_steps_eval = len(df_for_pred_eval); prediction_steps_eval = min(prediction_steps_eval, 1000) # Limit steps
                if prediction_steps_eval >= config.MIN_PREDICTION_STEPS:
                     predicted_states, true_states = multi_step_prediction(model, initial_sequence_eval, df_for_pred_eval, config.INPUT_SEQ_LEN, config.OUTPUT_SEQ_LEN, prediction_steps_eval, input_scaler, target_scaler, device)
                     if len(predicted_states) > 0:
                           model_mse = np.mean((predicted_states - true_states)**2)
                           # ... (Physics comparison generation remains the same) ...
                           physics_x0 = df_eval_segment.iloc[config.INPUT_SEQ_LEN][['theta', 'theta_dot']].values; physics_time_eval = df_for_pred_eval['time'].iloc[:prediction_steps_eval].values
                           if len(physics_time_eval) > 0:
                                physics_t_span = (physics_time_eval[0], physics_time_eval[-1]); physics_tau_values = df_for_pred_eval['tau'].iloc[:prediction_steps_eval].values
                                physics_time, physics_theta, physics_theta_dot = run_simulation(pendulum_eval, physics_t_span, config.DT, physics_x0, physics_tau_values, t_eval=physics_time_eval)
                                if len(physics_time) == len(physics_time_eval): physics_predictions = np.stack([physics_theta, physics_theta_dot], axis=1)
                           plot_filename = f"multistep_eval_{config.MODEL_TYPE}"
                           plot_multi_step_prediction(physics_time_eval, true_states, predicted_states, physics_predictions, config.MODEL_TYPE, config.FIGURES_DIR, filename_base=plot_filename)
                           print(f"多步预测图表 (评估段) 已保存到 {config.FIGURES_DIR}/{plot_filename}.png")
                else: print(f"评估段可用步数 ({prediction_steps_eval}) 不足。")
            else: print("无法为评估段创建序列。")
        else: print("无法生成用于多步预测评估的数据段。")
    except Exception as eval_msp_e: print(f"执行多步预测评估时出错: {eval_msp_e}"); import traceback; traceback.print_exc()
    eval_time = time.time() - eval_start_time; print(f"模型评估完成，耗时: {eval_time:.2f}秒")


    # --- Final Summary ---
    # ... (Summary printing remains the same) ...
    total_time = time.time() - start_time; print(f"\n项目执行完毕！总耗时: {total_time:.2f}秒") # ... etc ...
    # ... (Print performance and model summaries) ...


if __name__ == "__main__":
    run_experiment()
