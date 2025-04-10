# evaluate_saved_model.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import os
import time # Ensure time module is imported

# Import necessary modules from the project
import config
import utils
from model import get_model # Use the factory function
# Import functions needed for data generation fallback and physics comparison
from data_generation import PendulumSystem, generate_simulation_data, run_simulation
# Import functions needed for getting test data and potentially calculating MSE
from data_preprocessing import get_test_data_and_scalers # Only need this one now
# Import necessary evaluation functions (including the consolidated one)
from evaluation import evaluate_model, run_multi_scenario_evaluation, plot_multi_step_prediction # plot needed if called separately

def evaluate_best_model(model_info_path=config.MODEL_FINAL_PATH, # Path to load model type/params
                        model_path=config.MODEL_BEST_PATH,      # Path to load the best model's state_dict
                        data_path=config.VAL_DATA_FILE,         # <<<--- Default to loading VAL_DATA_FILE
                        limit_prediction_steps=None,            # Optional limit for prediction steps
                        scenarios_to_eval=config.SCENARIOS,     # Scenarios to evaluate
                        ics_to_eval=config.INITIAL_CONDITIONS_SPECIFIC # ICs to evaluate
                        # start_idx_in_test is removed, evaluation is now scenario/IC based
                        ):
    """
    Loads the best saved model state and evaluates its performance using
    multi-scenario multi-step prediction.
    If data_path (validation data file) is not found, it attempts to regenerate it.
    """
    print(f"--- Starting Evaluation of Saved Model ---")
    print(f"--- Attempting to load model state from: {model_path} ---")
    eval_overall_start_time = time.time()

    # --- Setup Device ---
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Load Data (Try loading VAL_DATA_FILE, fallback to generation) ---
    df_eval = None # Use df_eval to store the data used for evaluation context (primarily for scalers)
    try:
        print(f"Attempting to load evaluation context data from {data_path}...")
        df_eval = pd.read_csv(data_path)
        print(f"Successfully loaded evaluation context data, shape: {df_eval.shape}")
    except FileNotFoundError:
        print(f"Warning: Evaluation context data file '{data_path}' not found.")
        print("--- Attempting to regenerate validation data on-the-fly ---")
        # --- Data Generation Logic (Only for Validation Set) ---
        gen_start_time = time.time()
        val_dfs = []
        total_val_points = 0
        try:
            pendulum = PendulumSystem(m=config.PENDULUM_MASS, L=config.PENDULUM_LENGTH,
                                      g=config.GRAVITY, c=config.DAMPING_COEFF)
            print("Pendulum system initialized for data generation.")
            print(f"Generating validation data (T={config.T_SPAN_VAL}) for scenarios: {config.SCENARIOS}")
            for scenario_type in config.SCENARIOS:
                for i, x0 in enumerate(config.INITIAL_CONDITIONS_SPECIFIC):
                    df_val_single = generate_simulation_data(
                        pendulum, t_span=config.T_SPAN_VAL, dt=config.DT, x0=x0, torque_type=scenario_type
                    )
                    if not df_val_single.empty: val_dfs.append(df_val_single); total_val_points += len(df_val_single)
            if not val_dfs: raise RuntimeError("No validation data generated.")
            df_eval = pd.concat(val_dfs, ignore_index=True)
            gen_time = time.time() - gen_start_time
            print(f"--- Validation data regenerated successfully ({total_val_points} points), took {gen_time:.2f}s ---")
            try: df_eval.to_csv(data_path, index=False); print(f"Regenerated validation data saved to {data_path}.")
            except Exception as save_e: print(f"Warning: Could not save regenerated data. Error: {save_e}")
        except Exception as gen_e: print(f"Error during on-the-fly data generation: {gen_e}"); return
    except Exception as load_e: print(f"Error loading data from {data_path}: {load_e}"); return

    if df_eval is None or df_eval.empty: print("Error: DataFrame for evaluation context is empty. Aborting."); return

    # --- Load Model Info (to determine type/scaler paths) ---
    model_info = None
    model_type_for_eval = config.MODEL_TYPE # Default
    if os.path.exists(model_info_path):
        try:
            model_info = torch.load(model_info_path, map_location='cpu')
            loaded_type = model_info.get('model_type', config.MODEL_TYPE)
            if loaded_type: model_type_for_eval = loaded_type
            print(f"Loaded model info from {model_info_path}. Determined model type: {model_type_for_eval}")
        except Exception as e: print(f"Warning: Could not load model info file {model_info_path}. Error: {e}"); model_info = None
    else: print(f"Warning: Model info file {model_info_path} not found. Using MODEL_TYPE='{config.MODEL_TYPE}' from config.")

    # Determine correct target scaler path based on determined model type
    if model_type_for_eval.lower().startswith("delta"): target_scaler_path = config.DELTA_SCALER_PATH
    else: target_scaler_path = config.TARGET_SCALER_PATH
    print(f"Using target scaler path: {target_scaler_path}")

    # --- Load Scalers ---
    # Use get_test_data_and_scalers primarily to load the correct scalers
    # We don't necessarily need its returned X_val_scaled or df_for_pred anymore,
    # as the evaluation loop will generate segments as needed.
    # However, we DO need the fitted scalers.
    print("Loading scalers...")
    try:
        if not os.path.exists(config.INPUT_SCALER_PATH) or not os.path.exists(target_scaler_path):
             raise FileNotFoundError(f"Scaler file(s) not found: {config.INPUT_SCALER_PATH}, {target_scaler_path}")
        input_scaler = joblib.load(config.INPUT_SCALER_PATH)
        target_scaler = joblib.load(target_scaler_path)
        if not hasattr(input_scaler, 'scale_') or not hasattr(target_scaler, 'scale_'):
             raise ValueError("Loaded scalers appear to be not fitted.")
        print(f"Scalers loaded successfully ({config.INPUT_SCALER_PATH}, {target_scaler_path}).")
    except Exception as e: print(f"Error loading scalers: {e}"); return

    # --- Instantiate Model and Load State Dictionary ---
    model = None
    try:
        input_size = input_scaler.n_features_in_
        output_size = target_scaler.n_features_in_
        model = get_model(model_type=model_type_for_eval, input_size=input_size, output_size=output_size)
        print(f"Model architecture '{model_type_for_eval}' instantiated.")
        if not os.path.exists(model_path): raise FileNotFoundError(f"Model state file not found at {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device); model.eval()
        print(f"Successfully loaded model state from '{model_path}' to {device}")
    except Exception as e: print(f"Error instantiating or loading model: {e}"); return

    # --- Perform Evaluation using the consolidated function ---
    print("\n--- Calling Multi-Scenario Multi-Step Prediction Evaluation ---")
    # Ensure the evaluation function is available
    if hasattr(evaluation, 'run_multi_scenario_evaluation'):
        evaluation_results = evaluation.run_multi_scenario_evaluation(
            model=model,
            input_scaler=input_scaler,
            target_scaler=target_scaler,
            device=device,
            model_type_name=f"{model_type_for_eval} (Loaded)", # Pass descriptive name
            scenarios_to_eval=scenarios_to_eval,
            ics_to_eval=ics_to_eval,
            limit_prediction_steps=limit_prediction_steps,
            save_dir=config.FIGURES_DIR
        )
    else:
        print("Error: Consolidated evaluation function 'run_multi_scenario_evaluation' not found in evaluation.py.")
        evaluation_results = {} # Empty results

    # --- Final Timing ---
    eval_total_time = time.time() - eval_overall_start_time
    print(f"\nOverall evaluation script finished in {eval_total_time:.2f} seconds.")


if __name__ == "__main__":
    # Setup utilities
    utils.setup_logging_and_warnings()
    utils.setup_chinese_font()

    # --- Configuration for Evaluation ---
    # Specify which scenarios and ICs to evaluate (use indices from config)
    scenarios_to_run = config.SCENARIOS # Evaluate all scenarios
    ics_to_run = config.INITIAL_CONDITIONS_SPECIFIC # Evaluate all ICs
    # Example: Evaluate only 'sine' and 'random' scenarios for the first 2 ICs
    # scenarios_to_run = ["sine", "random"]
    # ics_to_run = config.INITIAL_CONDITIONS_SPECIFIC[:2]

    limit_steps = 1000 # Limit prediction length for speed (e.g., 1000 steps = 20 seconds)

    # Specify which trained model to evaluate by setting its type
    # This determines which _final.pth and _best.pth files are targeted
    evaluate_model_type = config.MODEL_TYPE # Evaluate the model type currently set in config
    # OR explicitly set it:
    # evaluate_model_type = "DeltaGRU"

    # Construct paths based on the chosen model type
    eval_model_basename = f'pendulum_{evaluate_model_type.lower()}'
    eval_model_info_path = os.path.join(config.MODELS_DIR, f'{eval_model_basename}_final.pth')
    eval_model_path = os.path.join(config.MODELS_DIR, f'{eval_model_basename}_best.pth')
    # Specify the validation data file path (used for fallback generation context if needed)
    eval_data_path = config.VAL_DATA_FILE

    print(f"--- Evaluating Model Type: {evaluate_model_type} ---")
    print(f"Using Best Model State: {eval_model_path}")
    print(f"Using Final Model Info: {eval_model_info_path}")
    print(f"Target Validation Data File: {eval_data_path} (regenerated if not found)")
    print(f"Scenarios to Evaluate: {scenarios_to_run}")
    print(f"Number of Initial Conditions per Scenario: {len(ics_to_run)}")
    print(f"Prediction Step Limit per run: {limit_steps if limit_steps else 'None'}")

    evaluate_best_model(
        model_info_path=eval_model_info_path,
        model_path=eval_model_path,
        data_path=eval_data_path, # Path for loading/saving regenerated val data
        limit_prediction_steps=limit_steps,
        scenarios_to_eval=scenarios_to_run,
        ics_to_eval=ics_to_run
        # start_idx_in_test is removed as evaluation is scenario/IC based now
    )
