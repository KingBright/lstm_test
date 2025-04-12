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
from model import get_model # Import model factory function
# *** Import function for generating the evaluation segment ***
from data_generation import PendulumSystem, generate_simulation_data, run_simulation # Keep run_simulation if physics comparison uses it
# Import sequence creation and evaluation functions
from data_preprocessing import create_sequences
from evaluation import multi_step_prediction, plot_multi_step_prediction # Removed evaluate_model as it's less relevant here

# Function to evaluate the saved model
def evaluate_best_model(model_info_path=config.MODEL_FINAL_PATH, # Use path from config
                        model_path=config.MODEL_BEST_PATH,       # Use path from config
                        input_scaler_path=config.INPUT_SCALER_PATH, # Use path from config
                        target_scaler_path=config.TARGET_SCALER_PATH,# Use path from config
                        limit_prediction_steps=None,
                        eval_duration=25.0 # Duration for the generated evaluation segment
                        ):
    """
    Loads the best saved model state and evaluates its multi-step prediction
    performance on a newly generated random segment of specified duration.
    Uses parameters stored in model_info_path for instantiation if possible.
    """
    print(f"--- Starting Standalone Evaluation of Saved Model ---")
    print(f"--- Loading model state from: {model_path} ---")
    print(f"--- Reading model info from: {model_info_path} ---")
    print(f"--- Loading input scaler from: {input_scaler_path} ---")
    print(f"--- Loading target scaler from: {target_scaler_path} ---")
    eval_overall_start_time = time.time()

    # --- Setup Device ---
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Load Scalers ---
    print("Loading scalers...")
    try:
        if not os.path.exists(input_scaler_path) or not os.path.exists(target_scaler_path):
             raise FileNotFoundError(f"Scaler file(s) not found: {input_scaler_path}, {target_scaler_path}")
        input_scaler = joblib.load(input_scaler_path)
        target_scaler = joblib.load(target_scaler_path)
        print(f"Scalers loaded successfully.")
        # Basic check if scalers seem fitted
        if not hasattr(input_scaler, 'scale_') and not hasattr(input_scaler, 'min_'): raise ValueError("Input scaler does not appear to be fitted.")
        if not hasattr(target_scaler, 'scale_') and not hasattr(target_scaler, 'min_'): raise ValueError("Target scaler does not appear to be fitted.")
    except Exception as e: print(f"Error loading scalers: {e}"); return

    # --- Load Model Info (to determine type/params and infer features) ---
    model_info = None
    model_type_for_eval = config.MODEL_TYPE # Default to current config
    loaded_params = {}
    use_sincos_from_info = config.USE_SINCOS_THETA # Default

    if os.path.exists(model_info_path):
        try:
            # Load model info (consider adding weights_only=True if appropriate, but info dict usually doesn't have large tensors)
            model_info = torch.load(model_info_path, map_location='cpu') # Load to CPU first
            model_type_for_eval = model_info.get('model_type', config.MODEL_TYPE)
            loaded_params['hidden_size'] = model_info.get('hidden_size')
            loaded_params['num_layers'] = model_info.get('num_layers')
            loaded_params['dense_units'] = model_info.get('dense_units')
            loaded_params['dropout_rate'] = model_info.get('dropout_rate')
            # Infer feature engineering setting
            saved_input_size = model_info.get('input_size')
            if isinstance(saved_input_size, int):
                 use_sincos_from_info = (saved_input_size == 4) # Infer based on input size (4 for sincos, 3 otherwise)
                 print(f"Inferred USE_SINCOS_THETA={use_sincos_from_info} from saved model info (input_size={saved_input_size}).")
            else:
                 print(f"Warning: Could not infer USE_SINCOS_THETA from model info. Assuming current config value: {use_sincos_from_info}.")

            print(f"Loaded model info. Determined model type: {model_type_for_eval}")
            print(f"  Loaded Params: { {k:v for k,v in loaded_params.items() if v is not None and v != 'unknown'} }")
        except Exception as e: print(f"Warning: Could not load model info file {model_info_path}. Error: {e}"); model_info = None
    else: print(f"Warning: Model info file {model_info_path} not found. Using current config defaults.")

    # --- Instantiate Model using Loaded/Config Parameters ---
    model = None
    try:
        # Determine input/output size from scalers (more reliable)
        input_size = input_scaler.n_features_in_
        output_size = target_scaler.n_features_in_ # Should be 2 (theta, theta_dot)

        # Get parameters for the determined model type
        # Use parameters from loaded info if available, otherwise use current config
        current_config_params = config.get_current_model_params() # Gets params for config.MODEL_TYPE
        eval_params_dict = current_config_params
        # If loaded type differs from current config, try to get params for loaded type
        if model_type_for_eval.lower() != config.MODEL_TYPE.lower():
             temp_eval_params = config.MODEL_PARAMS.get("defaults", {}).copy()
             base_key = model_type_for_eval.lower().replace("delta", "pure", 1).replace("seq2seq","pure",1)
             specific_params = config.MODEL_PARAMS.get(base_key)
             if specific_params:
                 temp_eval_params.update(specific_params)
                 eval_params_dict = temp_eval_params # Use params for the loaded type
                 print(f"Using parameters defined for loaded model type '{model_type_for_eval}'")
             else:
                 print(f"Warning: Params for loaded type '{model_type_for_eval}' not found in config. Using defaults for '{config.MODEL_TYPE}'.")

        # Determine final parameters, prioritizing loaded values if valid
        def get_final_param(key, loaded_val, config_dict_for_type, target_type):
            default_val = config_dict_for_type.get(key)
            final_val = default_val # Start with default
            if loaded_val is not None and not (isinstance(loaded_val, str) and loaded_val.lower() == 'unknown'):
                try:
                    final_val = target_type(loaded_val) # Try using loaded value
                except (ValueError, TypeError):
                    print(f"Warning: Loaded param '{key}' ({loaded_val}) has wrong type, using default: {default_val}")
                    final_val = default_val # Fallback to default if loaded is invalid type
            elif default_val is None:
                 raise ValueError(f"Cannot determine value for '{key}' - not in loaded info or config defaults.")
            return final_val

        hidden_size = get_final_param('hidden_size', loaded_params.get('hidden_size'), eval_params_dict, int)
        num_layers = get_final_param('num_layers', loaded_params.get('num_layers'), eval_params_dict, int)
        dense_units = get_final_param('dense_units', loaded_params.get('dense_units'), eval_params_dict, int)
        dropout = get_final_param('dropout_rate', loaded_params.get('dropout_rate'), eval_params_dict, float)
        if num_layers < 1: num_layers = 1 # Ensure at least one layer

        # Instantiate model using the factory function
        print(f"Instantiating model '{model_type_for_eval}' with: input={input_size}, hidden={hidden_size}, layers={num_layers}, output={output_size}, dense={dense_units}, dropout={dropout:.2f}")
        model = get_model(model_type=model_type_for_eval, # Use determined type
                          input_size=input_size,
                          output_size=output_size)
        print("Model instantiated.")

        # --- Load State Dictionary ---
        if not os.path.exists(model_path): raise FileNotFoundError(f"Model state file not found at {model_path}")
        print(f"Loading model state dictionary from {model_path} to {device}...")
        # Load state dict with map_location
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device); model.eval() # Ensure model is on device and in eval mode
        print(f"Successfully loaded model state.")

    except Exception as e: print(f"Error during model setup: {e}"); import traceback; traceback.print_exc(); return

    # --- Perform Evaluation (Single Segment) ---
    print("\n--- Multi-Step Prediction Evaluation (on newly generated segment) ---")
    eval_start_time = time.time()
    predicted_states, true_states, physics_predictions = None, None, None
    model_mse = np.nan

    try:
        # Generate a short segment with random IC and random torque for evaluation
        pendulum_eval = PendulumSystem()
        eval_x0 = [np.random.uniform(*config.THETA_RANGE), np.random.uniform(*config.THETA_DOT_RANGE)]
        print(f"Generating evaluation segment: duration={eval_duration}s, IC={np.round(eval_x0, 2)}, torque_type={config.TORQUE_TYPE}")
        df_eval_segment = generate_simulation_data( # Use the simpler generator
            pendulum_eval,
            t_span=(0, eval_duration),
            dt=config.DT,
            x0=eval_x0,
            torque_type=config.TORQUE_TYPE # Use torque type from config
        )

        # Check if enough data generated and prepare sequences
        min_required_len = config.INPUT_SEQ_LEN + config.MIN_PREDICTION_STEPS
        if not df_eval_segment.empty and len(df_eval_segment) >= min_required_len:
            print(f"Generated evaluation segment length: {len(df_eval_segment)}")
            # Use the feature engineering flag determined earlier (from info or config)
            eval_data_values = df_eval_segment[['theta', 'theta_dot', 'tau']].values
            X_eval, _ = create_sequences(
                eval_data_values,
                simulation_boundaries=None, # Single segment
                input_seq_len=config.INPUT_SEQ_LEN,
                output_seq_len=config.OUTPUT_SEQ_LEN,
                use_sincos=use_sincos_from_info # Use the determined flag
            )

            if len(X_eval) > 0:
                # Check feature consistency before scaling
                if input_scaler.n_features_in_ != X_eval.shape[2]:
                    raise ValueError(f"Input scaler expects {input_scaler.n_features_in_} features, but generated data sequence has {X_eval.shape[2]}. Check USE_SINCOS_THETA consistency ({use_sincos_from_info}).")

                # Scale the first sequence to use as initial input
                X_eval_scaled = input_scaler.transform(X_eval.reshape(-1, X_eval.shape[2])).reshape(X_eval.shape)
                initial_sequence_eval = X_eval_scaled[0] # Shape (input_seq_len, features)

                # df_for_pred starts after the initial sequence used
                df_for_pred_eval = df_eval_segment.iloc[config.INPUT_SEQ_LEN:].reset_index(drop=True)
                available_steps = len(df_for_pred_eval)
                prediction_steps_eval = available_steps
                if limit_prediction_steps is not None and limit_prediction_steps > 0:
                    prediction_steps_eval = min(prediction_steps_eval, limit_prediction_steps)
                print(f"Attempting multi-step prediction for {prediction_steps_eval} steps...")

                # Ensure enough steps remain for meaningful prediction
                if prediction_steps_eval >= config.MIN_PREDICTION_STEPS:
                     predicted_states, true_states = multi_step_prediction(
                         model, initial_sequence_eval, df_for_pred_eval,
                         input_seq_len=config.INPUT_SEQ_LEN,
                         output_seq_len=config.OUTPUT_SEQ_LEN,
                         prediction_steps=prediction_steps_eval,
                         input_scaler=input_scaler,
                         target_scaler=target_scaler, # Pass the correct target scaler
                         device=device
                     )

                     # Calculate MSE if prediction successful
                     if predicted_states is not None and len(predicted_states) > 0 and len(true_states) == len(predicted_states):
                         model_mse = np.mean((predicted_states - true_states)**2)
                         print(f"  Multi-step MSE (Evaluation Segment): {model_mse:.6f}")

                         # Generate physics comparison for this segment
                         print("  Generating physics comparison...")
                         try:
                             physics_pendulum_ref=PendulumSystem();
                             # Get IC for physics sim from the point after the initial sequence
                             physics_x0 = df_eval_segment.iloc[config.INPUT_SEQ_LEN][['theta', 'theta_dot']].values;
                             physics_time_eval = df_for_pred_eval['time'].iloc[:prediction_steps_eval].values
                             if len(physics_time_eval) > 0:
                                 physics_t_span=(physics_time_eval[0], physics_time_eval[-1]);
                                 # Use the torque values corresponding to the prediction steps
                                 physics_tau_values=df_for_pred_eval['tau'].iloc[:prediction_steps_eval].values
                                 # Run physics simulation
                                 _, physics_theta, physics_theta_dot = run_simulation(
                                     physics_pendulum_ref, physics_t_span, config.DT,
                                     physics_x0, physics_tau_values, t_eval=physics_time_eval
                                 )
                                 if len(physics_theta) == len(physics_time_eval):
                                     physics_predictions = np.stack([physics_theta, physics_theta_dot], axis=1)
                                     physics_mse = np.mean((physics_predictions - true_states)**2)
                                     print(f"  Physics Model MSE (Evaluation Segment): {physics_mse:.6f}")
                                 else:
                                     print("  Warning: Physics simulation output length mismatch.")
                         except Exception as phys_e: print(f"  Error generating physics comparison: {phys_e}")

                         # Plot results
                         plot_filename = f"standalone_eval_{model_type_for_eval}"
                         if len(physics_time_eval) > 0:
                             # Ensure all arrays for plotting have the same length
                             min_plot_len = min(len(physics_time_eval), len(true_states), len(predicted_states))
                             if physics_predictions is not None:
                                 min_plot_len = min(min_plot_len, len(physics_predictions))
                                 physics_predictions_plot = physics_predictions[:min_plot_len]
                             else:
                                 physics_predictions_plot = None

                             if min_plot_len > 0:
                                 plot_multi_step_prediction(
                                     physics_time_eval[:min_plot_len],
                                     true_states[:min_plot_len],
                                     predicted_states[:min_plot_len],
                                     physics_predictions_plot, # Use potentially truncated version
                                     f"{model_type_for_eval} (Standalone Eval)",
                                     config.FIGURES_DIR,
                                     filename_base=plot_filename
                                 )
                                 print(f"  Multi-step prediction plot saved to {config.FIGURES_DIR}/{plot_filename}.png")
                             else: print("  Warning: No valid steps to plot after length alignment.")
                         else: print("  Warning: Cannot plot, time vector for evaluation missing.")
                     else:
                         print("  Multi-step prediction did not return valid results.")
                else: print(f"评估段可用步数 ({prediction_steps_eval}) 不足 (最低要求: {config.MIN_PREDICTION_STEPS})。") # Available steps too few
            else: print("无法为评估段创建序列。") # Cannot create sequences
        else: print(f"无法生成足够长的用于多步预测评估的数据段 (需要 >={min_required_len} 点, 实际生成 {len(df_eval_segment)} 点)。") # Generated segment too short
    except Exception as eval_msp_e: print(f"执行多步预测评估时出错: {eval_msp_e}"); import traceback; traceback.print_exc()

    eval_time = time.time() - eval_start_time
    print(f"\nEvaluation finished in {eval_time:.2f} seconds.")
    print(f"Overall script time: {time.time() - eval_overall_start_time:.2f} seconds.")


if __name__ == "__main__":
    # *** Setup fonts for plotting ***
    utils.setup_logging_and_warnings()
    utils.setup_chinese_font()

    # --- Configuration for Standalone Evaluation ---
    limit_steps = 1000 # Limit prediction steps for plotting if needed
    eval_duration_secs = 25.0 # Duration of the simulation segment to generate

    # Paths are now taken directly from config in the function call
    print(f"--- Evaluating Model Based on Current Config ---")
    print(f"Model Type (from config): {config.MODEL_TYPE}")
    print(f"Using Best Model State: {config.MODEL_BEST_PATH}")
    print(f"Using Final Model Info: {config.MODEL_FINAL_PATH}")
    print(f"Using Input Scaler: {config.INPUT_SCALER_PATH}")
    print(f"Using Target Scaler: {config.TARGET_SCALER_PATH}")
    print(f"Prediction Step Limit: {limit_steps if limit_steps else 'None'}")
    print(f"Evaluation Segment Duration: {eval_duration_secs}s")

    evaluate_best_model(
        # Paths are now arguments with defaults from config
        limit_prediction_steps=limit_steps,
        eval_duration=eval_duration_secs
    )

    print("--- Standalone Evaluation Script Finished ---")

