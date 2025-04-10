# main_experiment.py

import time
import evaluation
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
from data_generation import PendulumSystem, generate_simulation_data # Keep run_simulation if needed by evaluation
# Import the NEW data prep function that takes separate train/val dfs
from data_preprocessing import prepare_train_val_data_from_dfs
from model import get_model # Use factory function
from training import train_model
# Import necessary evaluation functions (including the new one)
from evaluation import evaluate_model, plot_training_curves, run_multi_scenario_evaluation

def run_experiment():
    """
    Runs the complete workflow with separate generation for Train and Validation sets,
    and calls the consolidated multi-scenario evaluation function.
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

    # --- Step 2: Generate Separate Train and Validation Data ---
    print("步骤2: 分别为训练集和验证集生成仿真数据...")
    gen_start_time = time.time()
    train_dfs = []
    val_dfs = []
    scenario_plot_data = defaultdict(lambda: {'train': [], 'val': []}) # For plotting
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
        print("已加载预生成的数据文件。将跳过按场景绘图。")
        config.PLOT_SCENARIO_DATA = False # Disable plotting if loading
    else:
        print(f"生成新的训练和验证数据 (Force regenerate: {config.FORCE_REGENERATE_DATA})...")
        # ... (Data generation loop remains the same as previous version) ...
        print(f"场景: {config.SCENARIOS}")
        print(f"初始条件: {config.INITIAL_CONDITIONS_SPECIFIC}")
        print(f"训练时长: {config.T_SPAN_TRAIN}, 验证时长: {config.T_SPAN_VAL}")
        for scenario_type in config.SCENARIOS:
            print(f"  处理场景: {scenario_type}")
            scenario_train_dfs_for_plot = []
            scenario_val_dfs_for_plot = []
            for i, x0 in enumerate(config.INITIAL_CONDITIONS_SPECIFIC):
                df_train_single = generate_simulation_data(pendulum, t_span=config.T_SPAN_TRAIN, dt=config.DT, x0=x0, torque_type=scenario_type)
                if not df_train_single.empty: train_dfs.append(df_train_single); scenario_train_dfs_for_plot.append(df_train_single); total_train_points += len(df_train_single)
                else: print(f"    Warning: No train data generated for '{scenario_type}' with IC {x0}.")
                df_val_single = generate_simulation_data(pendulum, t_span=config.T_SPAN_VAL, dt=config.DT, x0=x0, torque_type=scenario_type)
                if not df_val_single.empty: val_dfs.append(df_val_single); scenario_val_dfs_for_plot.append(df_val_single); total_val_points += len(df_val_single)
                else: print(f"    Warning: No val data generated for '{scenario_type}' with IC {x0}.")
            scenario_plot_data[scenario_type]['train'] = scenario_train_dfs_for_plot
            scenario_plot_data[scenario_type]['val'] = scenario_val_dfs_for_plot
        if not train_dfs or not val_dfs: print("Error: 未能成功生成训练或验证数据。"); return
        df_train = pd.concat(train_dfs, ignore_index=True)
        df_val = pd.concat(val_dfs, ignore_index=True)
        try:
            os.makedirs(config.MODELS_DIR, exist_ok=True)
            df_train.to_csv(config.TRAIN_DATA_FILE, index=False)
            df_val.to_csv(config.VAL_DATA_FILE, index=False)
            print(f"训练数据已保存到 {config.TRAIN_DATA_FILE} ({len(df_train)} points)")
            print(f"验证数据已保存到 {config.VAL_DATA_FILE} ({len(df_val)} points)")
        except Exception as e: print(f"保存数据文件时出错: {e}")

    gen_time = time.time() - gen_start_time
    print(f"数据生成/加载完成，耗时: {gen_time:.2f}秒")
    print(f"总训练数据点: {total_train_points}, 总验证数据点: {total_val_points}")

    # --- Plotting Generated Data by Scenario (if generated) ---
    if config.PLOT_SCENARIO_DATA and scenario_plot_data:
        print("\n--- 绘制每个场景的训练/验证数据对比图 (仅角度) ---")
        # Ensure the plotting function exists in utils
        if hasattr(utils, 'plot_scenario_comparison'):
            for scenario_name, plot_data in scenario_plot_data.items():
                utils.plot_scenario_comparison(
                    scenario_name, plot_data['train'], plot_data['val'], config.FIGURES_DIR
                )
        else:
            print("Warning: Plotting function 'plot_scenario_comparison' not found in utils. Skipping scenario plots.")
    elif config.PLOT_SCENARIO_DATA:
        print("跳过按场景绘图，因为数据是加载的或生成失败。")

    # --- Step 3: Prepare DataLoaders from df_train and df_val ---
    print("\n步骤3: 从训练和验证 DataFrame 创建 DataLoaders...")
    data_prep_start_time = time.time()
    # Call the data prep function that takes df_train, df_val
    data_loaders_tuple = prepare_train_val_data_from_dfs(
        df_train, df_val, sequence_length=config.SEQUENCE_LENGTH, seed=config.SEED
    )
    if data_loaders_tuple is None or data_loaders_tuple[0] is None: print("Error: Failed to create datasets and loaders."); return
    train_loader, val_loader, input_scaler, target_scaler = data_loaders_tuple
    data_prep_time = time.time() - data_prep_start_time
    print(f"数据准备完成，耗时: {data_prep_time:.2f}秒")
    # ... (Print loader lengths) ...
    train_loader_len_check = len(train_loader) if train_loader else 0; val_loader_len_check = len(val_loader) if val_loader else 0
    print(f"Train loader batches: {train_loader_len_check}, Val loader batches: {val_loader_len_check}")
    if train_loader_len_check == 0: print("Error: Training loader is empty."); return

    # --- Step 4: Model Definition ---
    try: input_size = input_scaler.n_features_in_; output_size = target_scaler.n_features_in_
    except AttributeError: print("Error: Scalers invalid."); return
    try: model = get_model(model_type=config.MODEL_TYPE, input_size=input_size, output_size=output_size); print("Model created successfully."); print(model)
    except Exception as e: print(f"Error creating model: {e}"); return

    # --- Step 5: Model Training ---
    print("\n步骤5: 开始训练模型...")
    train_start_time = time.time()
    train_losses, val_losses, best_epoch = train_model(
        model, train_loader, val_loader, device=device,
        model_save_path=config.MODEL_BEST_PATH, final_model_info_path=config.MODEL_FINAL_PATH
    )
    train_time = time.time() - train_start_time
    print(f"模型训练完成，耗时: {train_time:.2f}秒")
    plot_training_curves(train_losses, val_losses, config.FIGURES_DIR)
    print("训练曲线已绘制并保存")
    if best_epoch == 0: print("Warning: Training did not yield improvement.")

    # --- Step 6: Model Evaluation (using consolidated function) ---
    print("\n步骤6: 开始在验证集上进行最终评估...")
    eval_start_time = time.time()

    # Load best model state
    model.to(device)
    if os.path.exists(config.MODEL_BEST_PATH) and best_epoch > 0:
        try: model.load_state_dict(torch.load(config.MODEL_BEST_PATH, map_location=device)); print(f"已加载最佳模型 (来自 Epoch {best_epoch})")
        except Exception as e: print(f"Warning: Failed to load best model state. Error: {e}")
    else: print("使用训练结束时的模型状态进行评估。")

    # --- Call the Consolidated Multi-Scenario Evaluation ---
    # Decide which scenarios/ICs to evaluate at the end of training
    # Maybe just a subset for speed, e.g., one scenario or one IC?
    scenarios_for_final_eval = config.SCENARIOS # Evaluate all scenarios
    ics_for_final_eval = config.INITIAL_CONDITIONS_SPECIFIC # Evaluate all ICs
    # Limit steps for faster feedback after training
    final_eval_limit_steps = 500 # e.g., predict 10 seconds

    # Ensure the evaluation function is available
    if hasattr(evaluation, 'run_multi_scenario_evaluation'):
        evaluation_results = evaluation.run_multi_scenario_evaluation(
            model=model,
            input_scaler=input_scaler,
            target_scaler=target_scaler,
            device=device,
            model_type_name=config.MODEL_TYPE, # Pass the name for labels
            scenarios_to_eval=scenarios_for_final_eval,
            ics_to_eval=ics_for_final_eval,
            limit_prediction_steps=final_eval_limit_steps,
            save_dir=config.FIGURES_DIR
        )
    else:
        print("Error: Consolidated evaluation function 'run_multi_scenario_evaluation' not found.")
        evaluation_results = {} # Empty results


    eval_time = time.time() - eval_start_time
    print(f"模型评估完成，耗时: {eval_time:.2f}秒")

    # --- Final Summary ---
    total_time = time.time() - start_time
    print(f"\n项目执行完毕！总耗时: {total_time:.2f}秒")
    print("\n===== 性能摘要 =====")
    print(f"数据生成/加载时间: {gen_time:.2f}秒")
    print(f"数据准备时间: {data_prep_time:.2f}秒")
    print(f"模型训练时间: {train_time:.2f}秒")
    print(f"模型评估时间: {eval_time:.2f}秒")
    print(f"总执行时间: {total_time:.2f}秒")

    print("\n===== 模型性能摘要 =====")
    # ... (Summary printing remains the same, using config.get_current_model_params() etc.) ...
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
    # avg_val_loss_final was calculated by evaluate_model if val_loader existed
    # print(f"验证集单步 MSE (最终模型): {avg_val_loss_final:.6f}") # Redundant if printed by evaluate_model

    print("\n--- 多步预测评估总结 (来自训练后运行) ---")
    if evaluation_results:
         valid_mses = [r['mse'] for r in evaluation_results.values() if np.isfinite(r['mse'])]
         if valid_mses: print(f"  所有成功评估运行的平均多步 MSE: {np.mean(valid_mses):.6f}")
         else: print("  未能成功计算任何多步预测的 MSE。")
    else: print("  未执行多步预测评估。")


    print("\n项目完成！")

if __name__ == "__main__":
    run_experiment()
