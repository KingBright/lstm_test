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
# We only need get_test_data_and_scalers now
from data_preprocessing import get_test_data_and_scalers, create_sequences
# Import necessary evaluation functions
from evaluation import evaluate_model, multi_step_prediction, plot_multi_step_prediction

def evaluate_best_model(model_info_path=config.MODEL_FINAL_PATH, # Path to load model type/params
                        model_path=config.MODEL_BEST_PATH,      # Path to load the best model's state_dict
                        data_path=config.VAL_DATA_FILE,         # <<<--- Default to loading VAL_DATA_FILE
                        limit_prediction_steps=None,            # Optional limit for prediction steps
                        start_idx_in_test=0):                   # Index within test set sequences to start prediction
    """
    Loads the best saved model state and evaluates its performance.
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
    df_eval = None # Use df_eval to store the data used for evaluation
    try:
        print(f"Attempting to load evaluation data from {data_path}...")
        df_eval = pd.read_csv(data_path)
        print(f"Successfully loaded evaluation data, shape: {df_eval.shape}")
    except FileNotFoundError:
        print(f"Warning: Evaluation data file '{data_path}' not found.")
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
                # print(f"  Generating validation data for scenario: {scenario_type}") # Less verbose
                for i, x0 in enumerate(config.INITIAL_CONDITIONS_SPECIFIC):
                    df_val_single = generate_simulation_data(
                        pendulum,
                        t_span=config.T_SPAN_VAL, # Use validation duration
                        dt=config.DT,
                        x0=x0,
                        torque_type=scenario_type
                    )
                    if not df_val_single.empty:
                        val_dfs.append(df_val_single)
                        total_val_points += len(df_val_single)

            if not val_dfs: raise RuntimeError("No validation data generated.")

            # Combine into a single DataFrame
            df_eval = pd.concat(val_dfs, ignore_index=True)
            gen_time = time.time() - gen_start_time
            print(f"--- Validation data regenerated successfully ({total_val_points} points), took {gen_time:.2f}s ---")
            try: # Attempt to save for next time
                 df_eval.to_csv(data_path, index=False)
                 print(f"Regenerated validation data saved to {data_path} for future use.")
            except Exception as save_e: print(f"Warning: Could not save regenerated data. Error: {save_e}")

        except Exception as gen_e: print(f"Error during on-the-fly data generation: {gen_e}"); return
        # --- End of Data Generation Logic ---
    except Exception as load_e: print(f"Error loading data from {data_path}: {load_e}"); return

    if df_eval is None or df_eval.empty: print("Error: DataFrame for evaluation is empty. Aborting."); return

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
    if model_type_for_eval.lower().startswith("delta"):
        target_scaler_path = config.DELTA_SCALER_PATH
    else:
        target_scaler_path = config.TARGET_SCALER_PATH
    print(f"Using target scaler path: {target_scaler_path}")

    # --- Get Evaluation Data Slice and Load Scalers ---
    # This function now loads the specified scalers and processes df_eval
    X_eval_scaled, df_for_pred, input_scaler, target_scaler = get_test_data_and_scalers(
        df_eval, # Use the loaded or generated validation dataframe
        sequence_length=config.SEQUENCE_LENGTH,
        # val_split is no longer needed here, function uses the whole df_eval
        input_scaler_path=config.INPUT_SCALER_PATH,
        target_scaler_path=target_scaler_path # Load the correct target scaler
    )

    if X_eval_scaled is None or df_for_pred is None or input_scaler is None or target_scaler is None:
        print("Failed to process evaluation data or load scalers. Aborting.")
        return

    # --- Instantiate Model and Load State Dictionary ---
    model = None
    try:
        input_size = input_scaler.n_features_in_
        output_size = target_scaler.n_features_in_ # Target scaler dictates output size
        # Use the factory function 'get_model' with the determined type
        model = get_model(model_type=model_type_for_eval,
                          input_size=input_size,
                          output_size=output_size)
        print(f"Model architecture '{model_type_for_eval}' instantiated.")
        # Load the state dictionary from model_path (usually the _best.pth file)
        if not os.path.exists(model_path): raise FileNotFoundError(f"Model state file not found at {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device); model.eval()
        print(f"Successfully loaded model state from '{model_path}' to {device}")
    except Exception as e: print(f"Error instantiating or loading model: {e}"); return

    # --- Perform Evaluation ---
    eval_start_time = time.time()

    # --- Multi-Step Prediction ---
    print("\n--- Multi-Step Prediction Evaluation ---")
    physics_predictions = None
    predicted_states = np.array([])
    true_states = np.array([])

    # Check if data is valid for prediction
    if X_eval_scaled is not None and not df_for_pred.empty and X_eval_scaled.shape[0] > 0:
        # Ensure start_idx_in_test is within the bounds of the sequences derived from evaluation data
        if 0 <= start_idx_in_test < len(X_eval_scaled):
            initial_sequence = X_eval_scaled[start_idx_in_test]

            # Adjust the prediction dataframe slice based on start_idx_in_test
            if start_idx_in_test < len(df_for_pred):
                 df_pred_slice = df_for_pred.iloc[start_idx_in_test:].reset_index(drop=True)
                 available_steps = len(df_pred_slice) # Predict until end of available data
            else:
                 df_pred_slice = pd.DataFrame(); available_steps = 0

            prediction_steps = available_steps
            if limit_prediction_steps is not None and limit_prediction_steps > 0:
                 prediction_steps = min(prediction_steps, limit_prediction_steps)
                 print(f"Limiting multi-step prediction to {prediction_steps} steps.")

            if prediction_steps >= config.MIN_PREDICTION_STEPS:
                 print(f"\nPerforming Multi-Step Prediction for {prediction_steps} steps...")
                 predicted_states, true_states = multi_step_prediction(
                     model, initial_sequence, df_pred_slice, config.SEQUENCE_LENGTH, prediction_steps,
                     input_scaler, target_scaler, device # Pass correct target scaler
                 )

                 # --- Generate Physics Comparison ---
                 if len(predicted_states) > 0 and len(true_states) == len(predicted_states):
                      print("Generating high-fidelity physics model prediction as reference...")
                      physics_pendulum_ref = PendulumSystem(m=config.PENDULUM_MASS, L=config.PENDULUM_LENGTH, g=config.GRAVITY, c=config.DAMPING_COEFF)
                      try:
                           if prediction_steps > 0 and len(df_pred_slice) >= prediction_steps:
                                physics_x0 = df_pred_slice.iloc[0][['theta', 'theta_dot']].values # State at start of prediction slice
                                physics_time_eval = df_pred_slice['time'].iloc[:prediction_steps].values
                                physics_t_span = (physics_time_eval[0], physics_time_eval[-1])
                                physics_tau_values = df_pred_slice['tau'].iloc[:prediction_steps].values
                                physics_dt = physics_time_eval[1] - physics_time_eval[0] if len(physics_time_eval) > 1 else config.DT

                                physics_time, physics_theta, physics_theta_dot = run_simulation(physics_pendulum_ref, physics_t_span, physics_dt, physics_x0, physics_tau_values, t_eval=physics_time_eval)

                                if len(physics_time) == len(physics_time_eval): physics_predictions = np.stack([physics_theta, physics_theta_dot], axis=1)
                                else: print("Warning: Physics simulation length mismatch."); physics_predictions = None
                           else: print("Warning: Not enough data for physics simulation."); physics_predictions = None

                           # --- Plotting ---
                           time_vector = physics_time_eval if 'physics_time_eval' in locals() and len(physics_time_eval)>0 else np.array([])
                           if len(time_vector) > 0:
                                final_plot_steps = min(len(time_vector), len(true_states), len(predicted_states))
                                plot_multi_step_prediction(
                                     time_vector[:final_plot_steps], true_states[:final_plot_steps], predicted_states[:final_plot_steps],
                                     physics_model_predictions=physics_predictions[:final_plot_steps] if physics_predictions is not None else None,
                                     model_name=f"{model_type_for_eval} (Loaded)", save_dir=config.FIGURES_DIR
                                )
                                print(f"Multi-step prediction plot saved to {config.FIGURES_DIR}")
                           else: print("Warning: Cannot plot multi-step results due to time vector issue.")

                      except Exception as e: print(f"Error during physics comparison or plotting: {e}"); physics_predictions = None
                 else: print("Multi-step prediction failed to produce results.")
            else: print(f"Insufficient data ({available_steps} steps) for multi-step prediction (min: {config.MIN_PREDICTION_STEPS}).")
        else: print(f"Skipping multi-step prediction (Invalid start index {start_idx_in_test}).")
    else: print("Skipping multi-step prediction (Evaluation data sequences not valid).")

    eval_time = time.time() - eval_start_time
    print(f"\nEvaluation finished in {eval_time:.2f} seconds.")
    print(f"Overall script time: {time.time() - eval_overall_start_time:.2f} seconds.")


if __name__ == "__main__":
    # Setup utilities
    utils.setup_logging_and_warnings()
    utils.setup_chinese_font()

    # --- Configuration for Evaluation ---
    start_index = 0 # Index within the validation set sequences to start prediction
    limit_steps = 1000 # Limit prediction length for faster evaluation (e.g., 1000 steps = 20 seconds)

    # Specify which trained model to evaluate by setting its type
    # This determines which _final.pth and _best.pth files are targeted
    evaluate_model_type = config.MODEL_TYPE # Evaluate the model type currently set in config
    # OR explicitly set it:
    # evaluate_model_type = "DeltaGRU"

    # Construct paths based on the chosen model type
    eval_model_basename = f'pendulum_{evaluate_model_type.lower()}'
    eval_model_info_path = os.path.join(config.MODELS_DIR, f'{eval_model_basename}_final.pth')
    eval_model_path = os.path.join(config.MODELS_DIR, f'{eval_model_basename}_best.pth')
    eval_data_path = config.VAL_DATA_FILE # Use the validation data file for evaluation

    print(f"--- Evaluating Model Type: {evaluate_model_type} ---")
    print(f"Using Best Model State: {eval_model_path}")
    print(f"Using Final Model Info: {eval_model_info_path}")
    print(f"Using Evaluation Data: {eval_data_path}")
    print(f"Prediction Start Index (in derived validation sequences): {start_index}")
    print(f"Prediction Step Limit: {limit_steps if limit_steps else 'None'}")

    evaluate_best_model(
        model_info_path=eval_model_info_path,
        model_path=eval_model_path,
        data_path=eval_data_path, # Pass the validation data path
        limit_prediction_steps=limit_steps,
        start_idx_in_test=start_index
    )
