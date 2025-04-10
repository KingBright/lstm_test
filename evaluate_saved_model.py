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
from model import get_model, PureLSTM, PureGRU # Import model classes AND factory function
# Import functions needed for data generation fallback and physics comparison
from data_generation import PendulumSystem, generate_simulation_data, run_simulation
# Import sequence creation
from data_preprocessing import create_sequences
# Import necessary evaluation functions - basic ones now
from evaluation import evaluate_model, multi_step_prediction, plot_multi_step_prediction

# Modified function signature - removed data_path, added eval_duration
def evaluate_best_model(model_info_path=config.MODEL_FINAL_PATH,
                        model_path=config.MODEL_BEST_PATH,
                        limit_prediction_steps=None,
                        eval_duration=10.0 # <<<--- Default evaluation duration (e.g., 10 seconds)
                        # scenarios_to_eval and ics_to_eval removed
                        ):
    """
    Loads the best saved model state and evaluates its multi-step prediction
    performance on a newly generated random segment of specified duration.
    Uses parameters stored in model_info_path for instantiation.
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

    # --- Data Generation Fallback (No longer loads external file for main eval data) ---
    # Data segment for evaluation will be generated on-the-fly later.
    # We only need to load scalers first.

    # --- Load Model Info (to determine type/params) ---
    model_info = None
    model_type_for_eval = config.MODEL_TYPE # Default
    loaded_params = {} # Store params loaded from file
    use_sincos_from_info = config.USE_SINCOS_THETA # Default to current config

    if os.path.exists(model_info_path):
        try:
            # Add weights_only=True for security
            model_info = torch.load(model_info_path, map_location='cpu', weights_only=True)
            model_type_for_eval = model_info.get('model_type', config.MODEL_TYPE) # Prioritize type from file
            # Load architecture parameters
            loaded_params['hidden_size'] = model_info.get('hidden_size')
            loaded_params['num_layers'] = model_info.get('num_layers')
            loaded_params['dense_units'] = model_info.get('dense_units')
            loaded_params['dropout_rate'] = model_info.get('dropout_rate')
            # Infer feature engineering setting from saved input size if possible
            saved_input_size = model_info.get('input_size')
            if isinstance(saved_input_size, int):
                 use_sincos_from_info = (saved_input_size == 4)
                 print(f"Inferred USE_SINCOS_THETA={use_sincos_from_info} from saved model info (input_size={saved_input_size}).")
            else:
                 print(f"Warning: Could not infer USE_SINCOS_THETA from model info. Assuming current config value: {use_sincos_from_info}.")

            print(f"Loaded model info. Determined model type: {model_type_for_eval}")
            print(f"  Loaded Params: { {k:v for k,v in loaded_params.items() if v is not None and v != 'unknown'} }")
        except Exception as e: print(f"Warning: Could not load model info file {model_info_path}. Error: {e}"); model_info = None
    else: print(f"Warning: Model info file {model_info_path} not found.")

    # --- Load Scalers ---
    # Determine expected scaler paths based on model type and inferred feature setting
    expected_target_scaler_path = ""
    eval_model_basename_for_scaler = f'pendulum_{model_type_for_eval.lower()}'
    if use_sincos_from_info: eval_model_basename_for_scaler += "_sincos"
    # Add suffix based on data generation strategy (e.g., _shortsim or _longtraj)
    # Assuming the model was trained with the corresponding data strategy set in config
    # This requires the saved model name convention to be consistent
    # Let's try to infer from the model_path if possible, otherwise use current config's basename convention
    if "_shortsim" in config.MODEL_BASENAME: eval_model_basename_for_scaler += "_shortsim"
    elif "_longtraj" in config.MODEL_BASENAME: eval_model_basename_for_scaler += "_longtraj"


    if model_type_for_eval.lower().startswith("delta"):
        expected_target_scaler_path = os.path.join(config.MODELS_DIR, f'{eval_model_basename_for_scaler}_delta_scaler.pkl')
    else: # Includes Seq2Seq absolute state prediction
        expected_target_scaler_path = os.path.join(config.MODELS_DIR, f'{eval_model_basename_for_scaler}_output_scaler.pkl')
    print(f"Expecting target scaler path: {expected_target_scaler_path}")

    expected_input_scaler_filename = f'{eval_model_basename_for_scaler}_input_scaler.pkl'
    expected_input_scaler_path = os.path.join(config.MODELS_DIR, expected_input_scaler_filename)
    print(f"Expecting input scaler path: {expected_input_scaler_path}")

    print("Loading scalers...")
    try:
        if not os.path.exists(expected_input_scaler_path) or not os.path.exists(expected_target_scaler_path):
             raise FileNotFoundError(f"Scaler file(s) not found: {expected_input_scaler_path}, {expected_target_scaler_path}")
        input_scaler = joblib.load(expected_input_scaler_path)
        target_scaler = joblib.load(expected_target_scaler_path) # Load the expected target scaler
        print(f"Scalers loaded ({expected_input_scaler_path}, {expected_target_scaler_path}).")
        if not hasattr(input_scaler, 'scale_') and not hasattr(input_scaler, 'min_'): raise ValueError("Input scaler not fitted.")
        if not hasattr(target_scaler, 'scale_') and not hasattr(target_scaler, 'min_'): raise ValueError("Target scaler not fitted.")
    except Exception as e: print(f"Error loading scalers: {e}"); return

    # --- Instantiate Model using Loaded/Config Parameters ---
    model = None
    try:
        input_size = input_scaler.n_features_in_
        output_size = target_scaler.n_features_in_ # Per-step output size (2)

        # Get default params from config for the determined model type
        current_config_params = config.get_current_model_params() # Gets params for config.MODEL_TYPE
        eval_params_dict = current_config_params
        if model_type_for_eval.lower() != config.MODEL_TYPE.lower():
             temp_eval_params = config.MODEL_PARAMS.get("defaults", {}).copy(); base_key = model_type_for_eval.lower().replace("delta", "pure", 1).replace("seq2seq","pure",1); specific_params = config.MODEL_PARAMS.get(base_key)
             if specific_params: temp_eval_params.update(specific_params)
             else: print(f"Warning: Params for loaded type '{model_type_for_eval}' not in config.")
             for key, value in config.MODEL_PARAMS.get("defaults", {}).items():
                 if key not in temp_eval_params: temp_eval_params[key] = value
             eval_params_dict = temp_eval_params

        # Determine final parameters: Use loaded value if valid, otherwise use value from eval_params_dict
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

        # Instantiate the correct model class
        model_type_key = model_type_for_eval.lower()
        if model_type_key.startswith("purelstm") or model_type_key.startswith("deltalstm") or model_type_key.startswith("seq2seqlstm"): model_class = PureLSTM
        elif model_type_key.startswith("puregru") or model_type_key.startswith("deltagru") or model_type_key.startswith("seq2seqgru"): model_class = PureGRU
        else: raise ValueError(f"Unknown model type: {model_type_for_eval}")

        print(f"Instantiating {model_class.__name__} with: input={input_size}, hidden={hidden_size}, layers={num_layers}, output={output_size}, dense={dense_units}, dropout={dropout}")
        model = model_class(input_size, hidden_size, num_layers, output_size, dense_units, dropout)
        print("Model instantiated.")

        # --- Load State Dictionary ---
        if not os.path.exists(model_path): raise FileNotFoundError(f"Model state file not found at {model_path}")
        # Add weights_only=True for security
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device); model.eval()
        print(f"Successfully loaded model state from '{model_path}' to {device}")

    except Exception as e: print(f"Error during model setup: {e}"); return

    # --- Perform Evaluation (Single Segment) ---
    print("\n--- Multi-Step Prediction Evaluation (on newly generated segment) ---")
    eval_start_time = time.time()
    predicted_states, true_states, physics_predictions = None, None, None
    model_mse = np.nan # Initialize MSE

    try:
        # Generate a short segment with random IC and random torque for evaluation
        pendulum = PendulumSystem() # Initialize pendulum
        eval_x0 = [np.random.uniform(*config.THETA_RANGE), np.random.uniform(*config.THETA_DOT_RANGE)]
        print(f"Generating evaluation segment: duration={eval_duration}s, IC={np.round(eval_x0, 2)}, torque={config.TORQUE_TYPE}")
        # Use the correct T_SPAN based on eval_duration
        t_span_eval = (0, eval_duration)
        df_eval_segment = generate_simulation_data(pendulum, t_span=t_span_eval, dt=config.DT, x0=eval_x0, torque_type=config.TORQUE_TYPE)

        # Check if enough data generated and prepare sequences
        # For Seq2Seq, need N input + K output + potentially more for prediction steps
        min_required_len = config.INPUT_SEQ_LEN + config.OUTPUT_SEQ_LEN # Min needed for one sequence pair
        if not df_eval_segment.empty and len(df_eval_segment) >= min_required_len:
            eval_data_values = df_eval_segment[['theta', 'theta_dot', 'tau']].values
            # Create sequences using the correct feature engineering flag for the loaded model
            X_eval, _ = create_sequences(eval_data_values, config.INPUT_SEQ_LEN, config.OUTPUT_SEQ_LEN,
                                         predict_delta=False, use_sincos=use_sincos_for_eval) # Use inferred flag

            if len(X_eval) > 0:
                # Check feature consistency before scaling
                if input_scaler.n_features_in_ != X_eval.shape[2]:
                    raise ValueError(f"Input scaler expects {input_scaler.n_features_in_} features, but generated data sequence has {X_eval.shape[2]}. Check USE_SINCOS_THETA consistency.")

                X_eval_scaled = input_scaler.transform(X_eval.reshape(-1, X_eval.shape[2])).reshape(X_eval.shape)
                initial_sequence_eval = X_eval_scaled[0]
                # df_for_pred starts after the initial sequence used
                df_for_pred_eval = df_eval_segment.iloc[config.INPUT_SEQ_LEN:].reset_index(drop=True)
                available_steps = len(df_for_pred_eval)
                prediction_steps_eval = available_steps
                if limit_prediction_steps is not None and limit_prediction_steps > 0:
                    prediction_steps_eval = min(prediction_steps_eval, limit_prediction_steps)

                # Ensure enough steps remain for meaningful prediction
                if prediction_steps_eval >= config.MIN_PREDICTION_STEPS:
                     print(f"Performing multi-step prediction for {prediction_steps_eval} steps...")
                     predicted_states, true_states = multi_step_prediction(
                         model, initial_sequence_eval, df_for_pred_eval,
                         input_seq_len=config.INPUT_SEQ_LEN, output_seq_len=config.OUTPUT_SEQ_LEN, # Pass N, K
                         prediction_steps=prediction_steps_eval,
                         input_scaler=input_scaler, target_scaler=target_scaler, device=device
                     )

                     # Calculate MSE if prediction successful
                     if len(predicted_states) > 0 and len(true_states) == len(predicted_states):
                         model_mse = np.mean((predicted_states - true_states)**2)
                         print(f"  Multi-step MSE (Evaluation Segment): {model_mse:.6f}")

                         # Generate physics comparison for this segment
                         print("  Generating physics comparison...")
                         try:
                             physics_pendulum_ref=PendulumSystem(); physics_x0 = df_eval_segment.iloc[config.INPUT_SEQ_LEN][['theta', 'theta_dot']].values; physics_time_eval = df_for_pred_eval['time'].iloc[:prediction_steps_eval].values
                             if len(physics_time_eval) > 0:
                                 physics_t_span=(physics_time_eval[0], physics_time_eval[-1]); physics_tau_values=df_for_pred_eval['tau'].iloc[:prediction_steps_eval].values
                                 physics_time, physics_theta, physics_theta_dot = run_simulation(physics_pendulum_ref, physics_t_span, config.DT, physics_x0, physics_tau_values, t_eval=physics_time_eval)
                                 if len(physics_time) == len(physics_time_eval): physics_predictions = np.stack([physics_theta, physics_theta_dot], axis=1)
                         except Exception as phys_e: print(f"  Error generating physics comparison: {phys_e}")

                         # Plot
                         plot_filename = f"multistep_eval_{model_type_for_eval}"
                         # Ensure physics_time_eval is defined before plotting
                         if 'physics_time_eval' in locals() and len(physics_time_eval) > 0:
                             final_plot_steps = min(len(physics_time_eval), len(true_states), len(predicted_states))
                             if final_plot_steps > 0:
                                 plot_multi_step_prediction(physics_time_eval[:final_plot_steps], true_states[:final_plot_steps], predicted_states[:final_plot_steps],
                                                            physics_predictions[:final_plot_steps] if physics_predictions is not None else None,
                                                            f"{model_type_for_eval} (Eval)", config.FIGURES_DIR, filename_base=plot_filename)
                                 print(f"  Multi-step prediction plot saved to {config.FIGURES_DIR}/{plot_filename}.png")
                             else: print("  Warning: No valid steps to plot.")
                         else: print("  Warning: Cannot plot, time vector for evaluation missing.")

                else: print(f"评估段可用步数 ({prediction_steps_eval}) 不足 (最低要求: {config.MIN_PREDICTION_STEPS})。")
            else: print("无法为评估段创建序列。")
        else: print(f"无法生成足够长的用于多步预测评估的数据段 (需要 >={min_required_len} 点, 实际生成 {len(df_eval_segment)} 点)。")
    except Exception as eval_msp_e: print(f"执行多步预测评估时出错: {eval_msp_e}"); import traceback; traceback.print_exc()

    eval_time = time.time() - eval_start_time
    print(f"\nEvaluation finished in {eval_time:.2f} seconds.")
    print(f"Overall script time: {time.time() - eval_overall_start_time:.2f} seconds.")


if __name__ == "__main__":
    utils.setup_logging_and_warnings(); utils.setup_chinese_font()
    # --- Configuration for Evaluation ---
    limit_steps = 1000
    eval_duration_secs = 10.0 # <<<--- Use shorter duration for evaluation segment

    # Specify which trained model to evaluate by setting its type from config
    evaluate_model_type = config.MODEL_TYPE
    eval_model_basename = f'pendulum_{evaluate_model_type.lower()}'
    # Determine if the model likely used sin/cos features based on current config for path construction
    use_sincos_in_name = config.USE_SINCOS_THETA # Assume matches config for path finding
    # TODO: Ideally, infer use_sincos_in_name from loaded model_info if available for robustness
    if use_sincos_in_name: eval_model_basename += "_sincos"
    # Add suffix based on data generation strategy if needed (e.g., _longtraj)
    if "_longtraj" in config.MODEL_BASENAME: eval_model_basename += "_longtraj"
    elif "_shortsim" in config.MODEL_BASENAME: eval_model_basename += "_shortsim"


    eval_model_info_path = os.path.join(config.MODELS_DIR, f'{eval_model_basename}_final.pth')
    eval_model_path = os.path.join(config.MODELS_DIR, f'{eval_model_basename}_best.pth')

    print(f"--- Evaluating Model Type: {evaluate_model_type} (Use SinCos: {use_sincos_in_name}) ---")
    print(f"Using Best Model State: {eval_model_path}")
    print(f"Using Final Model Info: {eval_model_info_path}")
    # print(f"Base Data File (for context/fallback): {eval_data_path}") # Removed data_path dependency
    print(f"Prediction Step Limit: {limit_steps if limit_steps else 'None'}")
    print(f"Evaluation Segment Duration: {eval_duration_secs}s")

    evaluate_best_model(
        model_info_path=eval_model_info_path,
        model_path=eval_model_path,
        # data_path removed
        limit_prediction_steps=limit_steps,
        eval_duration=eval_duration_secs # Pass duration for segment generation
        # scenarios_to_eval and ics_to_eval removed
    )
