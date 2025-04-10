# evaluate_saved_model.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import os
import time

# Import necessary modules from the project
import config
import utils
from model import get_model, PureLSTM, PureGRU
from data_generation import PendulumSystem, generate_simulation_data, run_simulation
from data_preprocessing import get_test_data_and_scalers
# Import necessary evaluation functions
# Ensure run_multi_scenario_evaluation is imported
from evaluation import evaluate_model, run_multi_scenario_evaluation

def evaluate_best_model(model_info_path=config.MODEL_FINAL_PATH,
                        model_path=config.MODEL_BEST_PATH,
                        data_path=config.VAL_DATA_FILE,
                        limit_prediction_steps=None,
                        scenarios_to_eval=config.SCENARIOS,
                        ics_to_eval=config.INITIAL_CONDITIONS_SPECIFIC
                        ):
    """
    Loads the best saved model state and evaluates its performance using
    multi-scenario multi-step prediction.
    Uses parameters stored in model_info_path for instantiation.
    If data_path (validation data file) is not found, it attempts to regenerate it.
    """
    print(f"--- Starting Evaluation of Saved Model ---")
    print(f"--- Loading model state from: {model_path} ---")
    print(f"--- Reading model info from: {model_info_path} ---")
    eval_overall_start_time = time.time()

    # --- Setup Device ---
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Load Data (Try loading VAL_DATA_FILE, fallback to generation) ---
    # ... (Data loading/generation logic remains the same) ...
    df_eval = None
    try:
        print(f"Attempting to load evaluation context data from {data_path}...")
        df_eval = pd.read_csv(data_path)
        print(f"Successfully loaded evaluation context data, shape: {df_eval.shape}")
    except FileNotFoundError:
        print(f"Warning: Evaluation context data file '{data_path}' not found. Regenerating...")
        gen_start_time = time.time(); val_dfs = []; total_val_points = 0
        try:
            pendulum = PendulumSystem(); print("Pendulum system initialized.")
            for scenario_type in config.SCENARIOS:
                for i, x0 in enumerate(config.INITIAL_CONDITIONS_SPECIFIC):
                    df_val_single = generate_simulation_data(pendulum, t_span=config.T_SPAN_VAL, dt=config.DT, x0=x0, torque_type=scenario_type)
                    if not df_val_single.empty: val_dfs.append(df_val_single); total_val_points += len(df_val_single)
            if not val_dfs: raise RuntimeError("No validation data generated.")
            df_eval = pd.concat(val_dfs, ignore_index=True); gen_time = time.time() - gen_start_time
            print(f"Validation data regenerated ({total_val_points} points), took {gen_time:.2f}s")
            try: df_eval.to_csv(data_path, index=False); print(f"Saved to {data_path}.")
            except Exception as save_e: print(f"Warning: Could not save regenerated data: {save_e}")
        except Exception as gen_e: print(f"Error during data generation: {gen_e}"); return
    except Exception as load_e: print(f"Error loading data: {load_e}"); return
    if df_eval is None or df_eval.empty: print("Error: DataFrame empty. Aborting."); return


    # --- Load Model Info and Determine Parameters ---
    # ... (Logic remains the same) ...
    model_info = None; model_type_for_eval = config.MODEL_TYPE; loaded_params = {}
    if os.path.exists(model_info_path):
        try:
            model_info = torch.load(model_info_path, map_location='cpu')
            model_type_for_eval = model_info.get('model_type', config.MODEL_TYPE)
            loaded_params['hidden_size'] = model_info.get('hidden_size'); loaded_params['num_layers'] = model_info.get('num_layers'); loaded_params['dense_units'] = model_info.get('dense_units'); loaded_params['dropout_rate'] = model_info.get('dropout_rate')
            print(f"Loaded model info from {model_info_path}. Determined model type: {model_type_for_eval}")
            print(f"  Loaded Params: { {k:v for k,v in loaded_params.items() if v is not None} }")
        except Exception as e: print(f"Warning: Could not load model info file {model_info_path}. Error: {e}"); model_info = None
    else: print(f"Warning: Model info file {model_info_path} not found.")


    # --- Load Scalers ---
    # ... (Logic remains the same, using config.TARGET_SCALER_PATH) ...
    target_scaler_path_to_load = config.TARGET_SCALER_PATH
    if model_type_for_eval.lower() != config.MODEL_TYPE.lower(): print(f"Warning: Model type for eval ('{model_type_for_eval}') differs from current config ('{config.MODEL_TYPE}'). Ensure scaler path '{target_scaler_path_to_load}' is correct.")
    print(f"Attempting to load target scaler from: {target_scaler_path_to_load}")
    print("Loading scalers...")
    try:
        if not os.path.exists(config.INPUT_SCALER_PATH) or not os.path.exists(target_scaler_path_to_load): raise FileNotFoundError(f"Scaler file(s) not found: {config.INPUT_SCALER_PATH}, {target_scaler_path_to_load}")
        input_scaler = joblib.load(config.INPUT_SCALER_PATH); target_scaler = joblib.load(target_scaler_path_to_load)
        print(f"Scalers loaded ({config.INPUT_SCALER_PATH}, {target_scaler_path_to_load}).")
        if not hasattr(input_scaler, 'scale_') and not hasattr(input_scaler, 'min_'): raise ValueError("Input scaler not fitted.")
        if not hasattr(target_scaler, 'scale_') and not hasattr(target_scaler, 'min_'): raise ValueError("Target scaler not fitted.")
    except Exception as e: print(f"Error loading scalers: {e}"); return


    # --- Instantiate Model using Loaded/Config Parameters ---
    # ... (Logic remains the same, using get_final_param helper) ...
    model = None
    try:
        input_size = input_scaler.n_features_in_; output_size = target_scaler.n_features_in_
        current_config_params = config.get_current_model_params()
        eval_params_dict = current_config_params
        if model_type_for_eval.lower() != config.MODEL_TYPE.lower():
             temp_eval_params = config.MODEL_PARAMS.get("defaults", {}).copy(); base_key = model_type_for_eval.lower().replace("delta", "pure", 1); specific_params = config.MODEL_PARAMS.get(base_key)
             if specific_params: temp_eval_params.update(specific_params)
             else: print(f"Warning: Params for loaded type '{model_type_for_eval}' not in config.")
             for key, value in config.MODEL_PARAMS.get("defaults", {}).items():
                 if key not in temp_eval_params: temp_eval_params[key] = value
             eval_params_dict = temp_eval_params
        def get_final_param(key, loaded_val, config_dict_for_type, target_type):
            default_val = config_dict_for_type.get(key)
            if loaded_val is not None and not (isinstance(loaded_val, str) and loaded_val.lower() == 'unknown'):
                try: return target_type(loaded_val)
                except (ValueError, TypeError): pass
            if default_val is not None: return target_type(default_val)
            else: raise ValueError(f"Cannot determine value for '{key}'")
        hidden_size = get_final_param('hidden_size', loaded_params.get('hidden_size'), eval_params_dict, int)
        num_layers = get_final_param('num_layers', loaded_params.get('num_layers'), eval_params_dict, int)
        dense_units = get_final_param('dense_units', loaded_params.get('dense_units'), eval_params_dict, int)
        dropout = get_final_param('dropout_rate', loaded_params.get('dropout_rate'), eval_params_dict, float)
        if num_layers < 1: num_layers = 1
        model_type_key = model_type_for_eval.lower()
        if model_type_key.startswith("purelstm") or model_type_key.startswith("deltalstm"): model_class = PureLSTM
        elif model_type_key.startswith("puregru") or model_type_key.startswith("deltagru"): model_class = PureGRU
        else: raise ValueError(f"Unknown model type: {model_type_for_eval}")
        print(f"Instantiating {model_class.__name__} with: input={input_size}, hidden={hidden_size}, layers={num_layers}, output={output_size}, dense={dense_units}, dropout={dropout}")
        model = model_class(input_size, hidden_size, num_layers, output_size, dense_units, dropout)
        print("Model instantiated.")
        if not os.path.exists(model_path): raise FileNotFoundError(f"Model state file not found at {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device); model.eval()
        print(f"Successfully loaded model state from '{model_path}' to {device}")
    except Exception as e: print(f"Error during model setup: {e}"); return

    # --- Perform Evaluation using the consolidated function ---
    print("\n--- Calling Multi-Scenario Multi-Step Prediction Evaluation ---")
    # VVVVVV REMOVE the hasattr check VVVVVV
    try:
        # Directly call the imported function
        evaluation_results = run_multi_scenario_evaluation(
            model=model,
            input_scaler=input_scaler,
            target_scaler=target_scaler, # Pass the loaded target scaler
            device=device,
            model_type_name=f"{model_type_for_eval} (Loaded)", # Pass descriptive name
            scenarios_to_eval=scenarios_to_eval,
            ics_to_eval=ics_to_eval,
            limit_prediction_steps=limit_prediction_steps,
            save_dir=config.FIGURES_DIR
        )
    except NameError:
         # This might happen if evaluation.py itself has an import error for the function
         print("Error: Function 'run_multi_scenario_evaluation' is not defined. Check imports in evaluation.py and here.")
         evaluation_results = {}
    except Exception as e:
         print(f"Error calling run_multi_scenario_evaluation: {e}")
         import traceback
         traceback.print_exc() # Print full traceback for debugging
         evaluation_results = {}
    # ^^^^^^ MODIFIED BLOCK ^^^^^^

    # --- Final Timing ---
    eval_total_time = time.time() - eval_overall_start_time
    print(f"\nOverall evaluation script finished in {eval_total_time:.2f} seconds.")


if __name__ == "__main__":
    utils.setup_logging_and_warnings(); utils.setup_chinese_font()
    # --- Configuration for Evaluation ---
    start_index = 0; limit_steps = 1000
    scenarios_to_run = config.SCENARIOS; ics_to_run = config.INITIAL_CONDITIONS_SPECIFIC
    evaluate_model_type = config.MODEL_TYPE
    eval_model_basename = f'pendulum_{evaluate_model_type.lower()}'
    eval_model_info_path = os.path.join(config.MODELS_DIR, f'{eval_model_basename}_final.pth')
    eval_model_path = os.path.join(config.MODELS_DIR, f'{eval_model_basename}_best.pth')
    eval_data_path = config.VAL_DATA_FILE
    # ... (Print config) ...
    print(f"--- Evaluating Model Type: {evaluate_model_type} ---") # ... etc
    evaluate_best_model(
        model_info_path=eval_model_info_path, model_path=eval_model_path, data_path=eval_data_path,
        limit_prediction_steps=limit_steps, scenarios_to_eval=scenarios_to_run, ics_to_eval=ics_to_run
    )

