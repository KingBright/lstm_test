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
from model import PureLSTM
from data_generation import PendulumSystem, run_simulation # For physics comparison
from data_preprocessing import get_test_data_and_scalers, prepare_data_for_training # Import BOTH functions now
from evaluation import evaluate_model, multi_step_prediction, plot_multi_step_prediction

def evaluate_best_model(model_path=config.MODEL_BEST_PATH,
                        data_path=config.DATA_FILE_EXTENDED,
                        limit_prediction_steps=None): # Optional limit
    """
    Loads the best saved model and evaluates its performance, including multi-step prediction.
    """
    print(f"--- Starting Evaluation of Saved Model: {model_path} ---")
    eval_overall_start_time = time.time()

    # --- Setup Device ---
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Load Data and Scalers ---
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded data from {data_path}, shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Use the helper function to get test data and scalers
    X_test_scaled, df_for_pred, input_scaler, output_scaler = get_test_data_and_scalers(
        df,
        sequence_length=config.SEQUENCE_LENGTH,
        test_split=config.TEST_SPLIT,
        input_scaler_path=config.INPUT_SCALER_PATH,
        output_scaler_path=config.OUTPUT_SCALER_PATH
    )

    if X_test_scaled is None or df_for_pred is None or input_scaler is None or output_scaler is None:
        print("Failed to load data or scalers. Aborting evaluation.")
        return

    # --- Load Model ---
    # Instantiate model - needs input/output size, ideally load from saved info
    # Let's try loading the final info file first, fallback to config
    model_info = None
    if os.path.exists(config.MODEL_FINAL_PATH):
         try:
              model_info = torch.load(config.MODEL_FINAL_PATH, map_location='cpu') # Load info to CPU first
              print(f"Loaded model info from {config.MODEL_FINAL_PATH}")
         except Exception as e:
              print(f"Warning: Could not load model info file {config.MODEL_FINAL_PATH}. Error: {e}")
              model_info = None # Fallback

    # Determine model parameters
    try:
        # Use scaler info for input/output size (should be reliable)
        input_size = input_scaler.n_features_in_
        output_size = output_scaler.n_features_in_

        # Get other params from model_info, falling back to config if needed or if value is 'unknown'
        def get_param(key, default_value, target_type):
            value = default_value
            if model_info:
                value = model_info.get(key, default_value)
            # If the loaded value is the fallback string, use the default from config instead
            if isinstance(value, str) and value.lower() == 'unknown':
                print(f"Warning: Loaded '{key}' was 'unknown', using default value {default_value} from config.")
                value = default_value
            # Ensure correct type
            try:
                return target_type(value)
            except (ValueError, TypeError):
                print(f"Warning: Could not convert loaded param '{key}' (value: {value}) to {target_type}. Using default {default_value}.")
                return default_value

        hidden_size = get_param('hidden_size', config.HIDDEN_SIZE, int)
        num_layers = get_param('num_layers', config.NUM_LAYERS, int)
        dense_units = get_param('dense_units', config.DENSE_UNITS, int)
        dropout = get_param('dropout', config.DROPOUT_RATE, float)

        # Ensure num_layers >= 1 for model instantiation
        if num_layers < 1:
            print(f"Warning: num_layers ({num_layers}) is less than 1, setting to 1.")
            num_layers = 1

        print(f"Instantiating PureLSTM with: input={input_size}, hidden={hidden_size}, layers={num_layers}, output={output_size}, dense={dense_units}, dropout={dropout}")
        model = PureLSTM(input_size, hidden_size, num_layers, output_size, dense_units, dropout)
        print("Model instantiated.")

    except AttributeError as ae:
        print(f"Error accessing scaler attributes (check if scalers loaded correctly): {ae}")
        return
    except Exception as e:
        print(f"Error determining model parameters or instantiating model: {e}")
        return

    # Load the best state dictionary
    if not os.path.exists(model_path):
        print(f"Error: Model state dictionary not found at {model_path}")
        return
    try:
        model.load_state_dict(torch.load(model_path, map_location=device)) # Load directly to device
        model.to(device)
        model.eval() # Set to evaluation mode
        print(f"Successfully loaded model state from {model_path} to {device}")
    except Exception as e:
        print(f"Error loading model state dictionary: {e}")
        return

    # --- Perform Evaluation ---
    eval_start_time = time.time()

    # 1. Evaluate Test MSE (Optional, requires recreating test DataLoader)
    #    This part is slightly redundant if main_experiment already did it,
    #    but useful for a standalone script. We need the test tensors.
    try:
         # Simple way: reconstruct test dataset/loader if needed
         _, _, _, _, _, _, X_test_tensor, y_test_tensor, _, test_loader, _, _ = \
             prepare_data_for_training(df, config.SEQUENCE_LENGTH, config.TEST_SPLIT) # Re-run prep to get loader

         if test_loader and len(test_loader.dataset) > 0:
              criterion = nn.MSELoss()
              print("\nCalculating Test Set MSE...")
              avg_test_loss = evaluate_model(model, test_loader, criterion, device)
         else:
              print("\nSkipping Test Set MSE calculation (no test data).")
              avg_test_loss = float('nan')
    except Exception as e:
         print(f"\nError during test set MSE calculation: {e}")
         avg_test_loss = float('nan')


    # 2. Multi-Step Prediction
    physics_predictions = None
    if X_test_scaled is not None and not df_for_pred.empty and X_test_scaled.shape[0] > 0:
        start_idx_in_test = 0 # Start from the beginning of the test set
        initial_sequence = X_test_scaled[start_idx_in_test]

        # Determine prediction steps
        available_steps = min(len(X_test_scaled) - start_idx_in_test, len(df_for_pred))
        prediction_steps = available_steps
        if limit_prediction_steps is not None and limit_prediction_steps < available_steps:
             prediction_steps = limit_prediction_steps
             print(f"Limiting multi-step prediction to {prediction_steps} steps.")

        if prediction_steps >= config.MIN_PREDICTION_STEPS:
             print(f"\nPerforming Multi-Step Prediction for {prediction_steps} steps...")
             predicted_states, true_states = multi_step_prediction(
                 model, initial_sequence, df_for_pred, config.SEQUENCE_LENGTH, prediction_steps,
                 input_scaler, output_scaler, device
             )

             # Generate Physics Comparison if prediction successful
             if len(predicted_states) > 0:
                  print("Generating high-fidelity physics model prediction as reference...")
                  physics_pendulum_ref = PendulumSystem(m=config.PENDULUM_MASS, L=config.PENDULUM_LENGTH,
                                                       g=config.GRAVITY, c=config.DAMPING_COEFF)
                  try:
                       physics_x0 = df_for_pred.iloc[0][['theta', 'theta_dot']].values
                       physics_time_eval = df_for_pred['time'].iloc[:prediction_steps].values
                       physics_t_span = (physics_time_eval[0], physics_time_eval[-1])
                       physics_tau_values = df_for_pred['tau'].iloc[:prediction_steps].values
                       physics_dt = physics_time_eval[1] - physics_time_eval[0] if len(physics_time_eval) > 1 else config.DT

                       physics_time, physics_theta, physics_theta_dot = run_simulation(
                            physics_pendulum_ref, physics_t_span, physics_dt, physics_x0, physics_tau_values, t_eval=physics_time_eval
                       )

                       if len(physics_time) == len(physics_time_eval):
                            physics_predictions = np.stack([physics_theta, physics_theta_dot], axis=1)
                       else:
                            print("Warning: Physics simulation length mismatch. Disabling comparison.")
                            physics_predictions = None

                       # Plot results
                       time_vector = physics_time_eval
                       plot_multi_step_prediction(
                            time_vector, true_states[:len(time_vector)], predicted_states[:len(time_vector)],
                            physics_model_predictions=physics_predictions,
                            model_name="PureLSTM (Loaded)", save_dir=config.FIGURES_DIR
                       )
                       print(f"Multi-step prediction plot saved to {config.FIGURES_DIR}")

                  except Exception as e:
                       print(f"Error during physics comparison generation: {e}")
                       physics_predictions = None
                       # Plot without physics
                       time_vector = df_for_pred['time'].iloc[:prediction_steps].values
                       plot_multi_step_prediction(time_vector, true_states, predicted_states, model_name="PureLSTM (Loaded)", save_dir=config.FIGURES_DIR)
                       print(f"Multi-step prediction plot (no physics comparison) saved to {config.FIGURES_DIR}")
             else:
                  print("Multi-step prediction failed to produce results.")
        else:
             print(f"Insufficient data ({available_steps} steps) for meaningful multi-step prediction (min: {config.MIN_PREDICTION_STEPS}).")
    else:
        print("Skipping multi-step prediction (no test data or dataframe slice).")

    eval_time = time.time() - eval_start_time
    print(f"\nEvaluation finished in {eval_time:.2f} seconds.")
    print(f"Overall script time: {time.time() - eval_overall_start_time:.2f} seconds.")


if __name__ == "__main__":
    # Setup utilities (optional for evaluation, mainly for fonts)
    utils.setup_logging_and_warnings()
    utils.setup_chinese_font()

    # You can change the model path or limit steps here if needed
    evaluate_best_model(
        model_path=config.MODEL_BEST_PATH,
        # limit_prediction_steps=1000 # Example: Limit prediction horizon
    )