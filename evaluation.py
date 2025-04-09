# evaluation.py

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import config # For default paths, maybe other eval settings
import utils # For safe_text in plotting

# Note: We assume scalers (input_scaler, output_scaler) are passed as arguments
# where needed, rather than reloading them here using joblib.

def plot_training_curves(train_losses, val_losses, save_dir=config.FIGURES_DIR):
    """
    Plots and saves the training and validation loss curves.

    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        save_dir (str): Directory to save the plot image.
    """
    if not train_losses or not val_losses:
        print("Warning: Empty loss lists provided. Skipping training curve plot.")
        return

    try:
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label='Training Loss')
        # Plot validation loss only if it contains valid numbers (not all NaN/inf)
        valid_val_losses = [l for l in val_losses if l is not None and np.isfinite(l)]
        if valid_val_losses:
             plt.plot(epochs, val_losses, label='Validation Loss') # Plot original list including potential NaNs if needed
             # Optionally plot only valid points: plt.plot(epochs[:len(valid_val_losses)], valid_val_losses, label='Validation Loss')

        plt.title('Model Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.grid(True)
        # Use log scale for y-axis if losses span multiple orders of magnitude
        if train_losses and max(train_losses) / max(1e-9, min(train_losses)) > 100: # Basic check
             plt.yscale('log')

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Training curves plot saved to {save_path}")

    except Exception as e:
        print(f"Error plotting training curves: {e}")

def evaluate_model(model, test_loader, criterion, device=None):
    """
    Evaluates the model on the test set and returns the average loss.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.
        criterion (nn.Module): The loss function (e.g., nn.MSELoss).
        device (torch.device): The device to run evaluation on.

    Returns:
        float: The average loss over the test set. Returns float('inf') if evaluation fails.
    """
    if device is None:
        if torch.backends.mps.is_available(): device = torch.device("mps")
        elif torch.cuda.is_available(): device = torch.device("cuda")
        else: device = torch.device("cpu")
    model.to(device) # Ensure model is on the correct device
    model.eval() # Set model to evaluation mode

    total_loss = 0.0
    batches = 0
    start_time = time.time()

    # Check if test_loader is valid
    try:
         if len(test_loader.dataset) == 0:
              print("Warning: Test dataset is empty. Cannot evaluate.")
              return float('inf')
    except TypeError:
         print("Error: Cannot determine test dataset length.")
         return float('inf')

    with torch.no_grad(): # Disable gradient calculations
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            try:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                # Check for NaN/Inf loss
                if torch.isnan(loss) or torch.isinf(loss):
                     print("Warning: NaN or Inf loss detected during evaluation.")
                     # Handle as needed - skip batch, return inf? Returning inf for now.
                     return float('inf')
                total_loss += loss.item()
                batches += 1
            except Exception as e:
                 print(f"Error during model evaluation forward pass: {e}")
                 return float('inf') # Indicate failure

    avg_loss = total_loss / batches if batches > 0 else float('inf')
    eval_time = time.time() - start_time

    if np.isfinite(avg_loss):
        print(f'Average Test Loss (MSE): {avg_loss:.6f}, Evaluation Time: {eval_time:.2f}s')
    else:
        print(f'Evaluation resulted in invalid average loss. Evaluation Time: {eval_time:.2f}s')

    return avg_loss


def multi_step_prediction(model, initial_sequence, df_pred, sequence_length, prediction_steps,
                         input_scaler, output_scaler, device=None):
    """
    Performs multi-step prediction using a trained LSTM model. (Pure LSTM version)

    Args:
        model (nn.Module): The trained PureLSTM model.
        initial_sequence (np.array): The starting sequence (scaled), shape (sequence_length, num_features).
        df_pred (pd.DataFrame): DataFrame containing future data (at least 'tau' and 'time')
                                covering the prediction horizon, starting from the step *after*
                                the initial sequence ends.
        sequence_length (int): Length of the input sequence.
        prediction_steps (int): Number of steps to predict.
        input_scaler (MinMaxScaler): Fitted scaler for input features.
        output_scaler (MinMaxScaler): Fitted scaler for output features.
        device (torch.device): Device to run predictions on.

    Returns:
        tuple: (predicted_states_original, true_states_original)
               predicted_states_original shape: (prediction_steps, output_size)
               true_states_original shape: (prediction_steps, output_size)
               Returns empty arrays if prediction fails.
    """
    if device is None:
        if torch.backends.mps.is_available(): device = torch.device("mps")
        elif torch.cuda.is_available(): device = torch.device("cuda")
        else: device = torch.device("cpu")
    model.to(device) # Ensure model is on correct device
    model.eval() # Set to evaluation mode

    # --- Input Validation ---
    if not isinstance(initial_sequence, np.ndarray) or initial_sequence.ndim != 2 or initial_sequence.shape[0] != sequence_length:
         print(f"Error: Initial sequence shape invalid. Expected ({sequence_length}, features), got {initial_sequence.shape}")
         return np.empty((0, output_scaler.n_features_in_)), np.empty((0, output_scaler.n_features_in_))
    if df_pred.empty or len(df_pred) < prediction_steps:
         print(f"Warning: Prediction dataframe 'df_pred' is empty or shorter ({len(df_pred)}) than prediction steps ({prediction_steps}). Adjusting steps.")
         prediction_steps = len(df_pred)
         if prediction_steps <= 0:
             print("Error: No data available for prediction timeframe.")
             return np.empty((0, output_scaler.n_features_in_)), np.empty((0, output_scaler.n_features_in_))
    if not hasattr(input_scaler, 'scale_') or not hasattr(output_scaler, 'scale_'):
         print("Error: Input or output scaler is not fitted. Cannot proceed.")
         return np.empty((0, output_scaler.n_features_in_)), np.empty((0, output_scaler.n_features_in_))


    current_sequence_np = initial_sequence.copy()
    predicted_states_list = [] # Store scaled predictions
    num_input_features = initial_sequence.shape[1]
    num_output_features = output_scaler.n_features_in_ # Get output size from scaler

    print(f"Running multi-step prediction for {prediction_steps} steps...")

    with torch.no_grad():
        for i in range(prediction_steps):
            # Prepare tensor for model input
            current_sequence_tensor = torch.tensor(
                current_sequence_np.reshape(1, sequence_length, num_input_features), # Add batch dim
                dtype=torch.float32
            ).to(device)

            # --- Prediction ---
            try:
                next_state_scaled_tensor = model(current_sequence_tensor)
                next_state_scaled_np = next_state_scaled_tensor.cpu().numpy()[0] # Shape (output_size,)

                # Basic check for NaN/Inf in prediction
                if not np.all(np.isfinite(next_state_scaled_np)):
                     print(f"Warning: NaN or Inf detected in model output at step {i}. Stopping prediction.")
                     break

            except Exception as e:
                print(f"Error during model forward pass at step {i}: {e}")
                break # Stop prediction on error

            predicted_states_list.append(next_state_scaled_np)

            # --- Prepare Input for Next Step ---
            # Get the known 'tau' for the *next* time step from the provided dataframe
            # The index 'i' corresponds to the prediction step, which aligns with df_pred index
            next_tau_original = df_pred.iloc[i]['tau']

            try:
                # Inverse scale the predicted state [theta, theta_dot]
                predicted_state_unscaled = output_scaler.inverse_transform(next_state_scaled_np.reshape(1, -1))[0]

                # Create the feature vector for the next input step [theta, theta_dot, tau] (unscaled)
                next_input_features_unscaled = np.zeros(num_input_features)
                # Ensure correct assignment based on expected input structure
                if num_input_features >= num_output_features:
                     next_input_features_unscaled[0:num_output_features] = predicted_state_unscaled
                else: # Should not happen if data prep was consistent
                     raise ValueError("Input features < Output features, cannot reconstruct input.")
                # Assume tau is the last feature if more features exist
                if num_input_features > num_output_features:
                     next_input_features_unscaled[-1] = next_tau_original
                elif num_input_features == num_output_features and 'tau' in df_pred.columns:
                      # This implies tau might not be the last feature, or structure mismatch
                      # Let's assume standard [theta, theta_dot, tau] -> num_input=3, num_output=2
                      if num_input_features == 3 and num_output_features == 2:
                           next_input_features_unscaled[2] = next_tau_original
                      else:
                           print("Warning: Ambiguous feature structure. Assuming tau is not used or appended incorrectly.")
                           # Fallback or error needed here depending on expected structure

                # Scale the complete feature vector for the next input sequence
                next_step_features_scaled = input_scaler.transform(next_input_features_unscaled.reshape(1, -1))[0]

            except ValueError as ve:
                 print(f"Error processing features for next step {i+1}: {ve}")
                 break
            except Exception as e:
                 print(f"Unexpected error preparing next input at step {i+1}: {e}")
                 break

            # Update the sequence buffer (roll window)
            current_sequence_np = np.append(
                current_sequence_np[1:], # Remove oldest step
                next_step_features_scaled.reshape(1, num_input_features), # Add newest step
                axis=0
            )
            # End of loop for prediction_steps

    # --- Process Results ---
    actual_prediction_steps = len(predicted_states_list) # Number of steps actually predicted
    if actual_prediction_steps == 0:
        print("No steps were successfully predicted.")
        return np.empty((0, num_output_features)), np.empty((0, num_output_features))

    # Inverse transform all collected predictions
    try:
        predicted_states_scaled_array = np.array(predicted_states_list)
        predicted_states_original = output_scaler.inverse_transform(predicted_states_scaled_array)
    except Exception as e:
        print(f"Error during final inverse transform of predictions: {e}")
        return np.empty((0, num_output_features)), np.empty((0, num_output_features))

    # Get the corresponding true states from the prediction dataframe
    true_states_original = df_pred.iloc[:actual_prediction_steps][['theta', 'theta_dot']].values

    # Final check for length consistency
    if len(predicted_states_original) != len(true_states_original):
        print("Warning: Mismatch between final predicted and true state lengths. Trimming.")
        min_len = min(len(predicted_states_original), len(true_states_original))
        predicted_states_original = predicted_states_original[:min_len]
        true_states_original = true_states_original[:min_len]

    # Calculate final MSE for the multi-step prediction
    if len(predicted_states_original) > 0:
        mse = np.mean((predicted_states_original - true_states_original)**2)
        print(f"Multi-step Prediction MSE for {actual_prediction_steps} steps: {mse:.6f}")
    else:
        print("Could not calculate multi-step MSE (no valid predictions).")


    return predicted_states_original, true_states_original


def plot_multi_step_prediction(time_vector, true_states, predicted_states,
                               physics_model_predictions=None, model_name="LSTM",
                               save_dir=config.FIGURES_DIR):
    """
    Plots the comparison of multi-step predictions against true values and
    optionally against a pure physics model prediction.

    Args:
        time_vector (np.array): Time points corresponding to the states.
        true_states (np.array): Array of true states (n_steps, n_outputs).
        predicted_states (np.array): Array of model's predicted states (n_steps, n_outputs).
        physics_model_predictions (np.array, optional): Array of physics model states.
        model_name (str): Name of the model for titles/labels.
        save_dir (str): Directory to save the plot images.
    """
    # --- Input Checks ---
    if len(predicted_states) == 0 or len(true_states) == 0 or len(time_vector) == 0:
        print("Warning: Empty data provided for multi-step plotting.")
        return
    min_len = len(predicted_states)
    if len(true_states) != min_len or len(time_vector) != min_len:
         print(f"Warning: Length mismatch in plot data. Using shortest length: {min_len}")
         true_states = true_states[:min_len]
         time_vector = time_vector[:min_len]
         if physics_model_predictions is not None:
              physics_model_predictions = physics_model_predictions[:min_len]
    if physics_model_predictions is not None and len(physics_model_predictions) != min_len:
         print("Warning: Physics prediction length mismatch. Disabling physics plot.")
         physics_model_predictions = None

    # --- Calculations ---
    theta_error = np.abs(predicted_states[:, 0] - true_states[:, 0])
    theta_dot_error = np.abs(predicted_states[:, 1] - true_states[:, 1])
    mean_theta_error = np.mean(theta_error)
    mean_theta_dot_error = np.mean(theta_dot_error)

    mean_physics_theta_error = np.nan
    mean_physics_theta_dot_error = np.nan
    if physics_model_predictions is not None:
        physics_theta_error = np.abs(physics_model_predictions[:, 0] - true_states[:, 0])
        physics_theta_dot_error = np.abs(physics_model_predictions[:, 1] - true_states[:, 1])
        mean_physics_theta_error = np.mean(physics_theta_error)
        mean_physics_theta_dot_error = np.mean(physics_theta_dot_error)

    # --- Plotting ---
    try:
        fig = plt.figure(figsize=(15, 12), dpi=300)
        gs = plt.GridSpec(3, 2, figure=fig)
        fig.suptitle(utils.safe_text(f'多步预测分析: {model_name} 模型', f'Multi-step Prediction: {model_name} Model'), fontsize=16, fontweight='bold')

        # Row 1: State Predictions
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time_vector, true_states[:, 0], 'g-', label=utils.safe_text('真实角度', 'True Theta'), linewidth=2)
        ax1.plot(time_vector, predicted_states[:, 0], 'r--', label=utils.safe_text(f'预测角度 ({model_name})', f'Predicted Theta ({model_name})'), linewidth=2)
        if physics_model_predictions is not None:
            ax1.plot(time_vector, physics_model_predictions[:, 0], 'b-.', label=utils.safe_text('纯物理模型', 'Physics Model'), linewidth=1.5, alpha=0.7)
        ax1.set_title(utils.safe_text('角度 (θ) 预测', 'Theta (θ) Prediction'), fontsize=14)
        ax1.set_ylabel(utils.safe_text('角度 (rad)', 'Angle (rad)'), fontsize=12)
        ax1.text(0.02, 0.02, utils.safe_text(f'{model_name} 平均误差: {mean_theta_error:.4f} rad', f'{model_name} Mean Error: {mean_theta_error:.4f} rad'), transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        ax1.legend(fontsize=10); ax1.grid(True)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time_vector, true_states[:, 1], 'g-', label=utils.safe_text('真实角速度', 'True Angular Vel.'), linewidth=2)
        ax2.plot(time_vector, predicted_states[:, 1], 'r--', label=utils.safe_text(f'预测角速度 ({model_name})', f'Predicted Angular Vel. ({model_name})'), linewidth=2)
        if physics_model_predictions is not None:
            ax2.plot(time_vector, physics_model_predictions[:, 1], 'b-.', label=utils.safe_text('纯物理模型', 'Physics Model'), linewidth=1.5, alpha=0.7)
        ax2.set_title(utils.safe_text('角速度 (θ̇) 预测', 'Angular Velocity (θ̇) Prediction'), fontsize=14)
        ax2.set_ylabel(utils.safe_text('角速度 (rad/s)', 'Angular Vel. (rad/s)'), fontsize=12)
        ax2.text(0.02, 0.02, utils.safe_text(f'{model_name} 平均误差: {mean_theta_dot_error:.4f} rad/s', f'{model_name} Mean Error: {mean_theta_dot_error:.4f} rad/s'), transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        ax2.legend(fontsize=10); ax2.grid(True)

        # Row 2: Error Analysis
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(time_vector, theta_error, 'r-', label=utils.safe_text(f'{model_name} 角度误差', f'{model_name} Theta Error'), linewidth=2)
        if physics_model_predictions is not None:
            ax3.plot(time_vector, physics_theta_error, 'b-.', label=utils.safe_text('物理模型误差', 'Physics Model Error'), linewidth=1.5, alpha=0.7)
            ax3.axhline(y=mean_physics_theta_error, color='b', linestyle=':', alpha=0.6, label=utils.safe_text(f'物理平均误差: {mean_physics_theta_error:.4f}', f'Physics Mean Err: {mean_physics_theta_error:.4f}'))
        ax3.set_title(utils.safe_text('角度预测误差', 'Theta Prediction Error'), fontsize=14)
        ax3.set_ylabel(utils.safe_text('|误差| (rad)', '|Error| (rad)'), fontsize=12)
        ax3.axhline(y=mean_theta_error, color='r', linestyle='--', alpha=0.7, label=utils.safe_text(f'{model_name} 平均误差: {mean_theta_error:.4f}', f'{model_name} Mean Err: {mean_theta_error:.4f}'))
        ax3.legend(fontsize=10); ax3.grid(True)

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(time_vector, theta_dot_error, 'r-', label=utils.safe_text(f'{model_name} 角速度误差', f'{model_name} Angular Vel. Error'), linewidth=2)
        if physics_model_predictions is not None:
            ax4.plot(time_vector, physics_theta_dot_error, 'b-.', label=utils.safe_text('物理模型误差', 'Physics Model Error'), linewidth=1.5, alpha=0.7)
            ax4.axhline(y=mean_physics_theta_dot_error, color='b', linestyle=':', alpha=0.6, label=utils.safe_text(f'物理平均误差: {mean_physics_theta_dot_error:.4f}', f'Physics Mean Err: {mean_physics_theta_dot_error:.4f}'))
        ax4.set_title(utils.safe_text('角速度预测误差', 'Angular Velocity Prediction Error'), fontsize=14)
        ax4.set_ylabel(utils.safe_text('|误差| (rad/s)', '|Error| (rad/s)'), fontsize=12)
        ax4.axhline(y=mean_theta_dot_error, color='r', linestyle='--', alpha=0.7, label=utils.safe_text(f'{model_name} 平均误差: {mean_theta_dot_error:.4f}', f'{model_name} Mean Err: {mean_theta_dot_error:.4f}'))
        ax4.legend(fontsize=10); ax4.grid(True)

        # Row 3: Phase Plot and Cumulative Error
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(true_states[:, 0], true_states[:, 1], 'g-', label=utils.safe_text('真实轨迹', 'True Trajectory'), linewidth=2, alpha=0.8)
        ax5.plot(predicted_states[:, 0], predicted_states[:, 1], 'r--', label=utils.safe_text(f'预测轨迹 ({model_name})', f'Predicted Trajectory ({model_name})'), linewidth=2, alpha=0.8)
        if physics_model_predictions is not None:
            ax5.plot(physics_model_predictions[:, 0], physics_model_predictions[:, 1], 'b-.', label=utils.safe_text('纯物理轨迹', 'Physics Trajectory'), linewidth=1.5, alpha=0.6)
        ax5.set_title(utils.safe_text('相位图: 角度 vs 角速度', 'Phase Plot'), fontsize=14)
        ax5.set_xlabel(utils.safe_text('角度 (rad)', 'Angle (rad)'), fontsize=12)
        ax5.set_ylabel(utils.safe_text('角速度 (rad/s)', 'Angular Vel. (rad/s)'), fontsize=12)
        ax5.legend(fontsize=10); ax5.grid(True)

        ax6 = fig.add_subplot(gs[2, 1])
        # Use np.arange for denominator to avoid potential division by zero if len is 1
        steps_axis = np.arange(1, len(theta_error) + 1)
        cum_error_theta = np.cumsum(theta_error) / steps_axis
        cum_error_theta_dot = np.cumsum(theta_dot_error) / steps_axis
        ax6.plot(time_vector, cum_error_theta, 'r-', label=utils.safe_text(f'累积角度误差 ({model_name})', f'Cum. Theta Error ({model_name})'), linewidth=2)
        ax6.plot(time_vector, cum_error_theta_dot, 'm-', label=utils.safe_text(f'累积角速度误差 ({model_name})', f'Cum. Angular Vel. Error ({model_name})'), linewidth=2)
        if physics_model_predictions is not None:
            cum_error_physics_theta = np.cumsum(physics_theta_error) / steps_axis
            cum_error_physics_theta_dot = np.cumsum(physics_theta_dot_error) / steps_axis
            ax6.plot(time_vector, cum_error_physics_theta, 'b--', label=utils.safe_text('物理累积角度误差', 'Physics Cum. Theta Error'), linewidth=1.5, alpha=0.7)
            ax6.plot(time_vector, cum_error_physics_theta_dot, 'c--', label=utils.safe_text('物理累积角速度误差', 'Physics Cum. Angular Vel. Error'), linewidth=1.5, alpha=0.7)
        ax6.set_title(utils.safe_text('累积平均误差', 'Cumulative Average Error'), fontsize=14)
        ax6.set_xlabel(utils.safe_text('时间 (s)', 'Time (s)'), fontsize=12)
        ax6.set_ylabel(utils.safe_text('累积平均误差', 'Cum. Avg. Error'), fontsize=12)
        ax6.legend(fontsize=10); ax6.grid(True); ax6.set_yscale('log')

        # Add common x-label and footer text
        for ax in [ax1, ax2, ax3, ax4]: ax.set_xlabel(utils.safe_text('时间 (s)', 'Time (s)'), fontsize=12)
        fig.text(0.5, 0.01, utils.safe_text(f'预测步数: {len(time_vector)}   时间范围: {time_vector[0]:.2f}s - {time_vector[-1]:.2f}s', f'Steps: {len(time_vector)} Range: {time_vector[0]:.2f}s - {time_vector[-1]:.2f}s'), ha='center', fontsize=12)

        # Layout and Save
        plt.tight_layout(rect=[0, 0.02, 1, 0.97])
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'multi_step_prediction.png')
        save_path_hq = os.path.join(save_dir, 'multi_step_prediction_hq.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_hq, dpi=600, bbox_inches='tight') # Save high quality version too
        plt.close()
        print(f"Multi-step prediction plot saved to {save_path}")

    except Exception as e:
        print(f"Error plotting multi-step prediction: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for plotting errors