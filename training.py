# training.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import os
import config # Import config for hyperparameters and paths
# Assuming model.py defines the model structure, but train_model receives the instantiated model
# from model import PureLSTM # Not strictly needed here, model is passed as argument

def train_model(model, train_loader, test_loader, num_epochs=config.NUM_EPOCHS,
                learning_rate=config.LEARNING_RATE, device=None,
                early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
                scheduler_factor=config.SCHEDULER_FACTOR,
                scheduler_patience=config.SCHEDULER_PATIENCE,
                weight_decay=config.WEIGHT_DECAY,
                model_save_path=config.MODEL_BEST_PATH,
                final_model_info_path=config.MODEL_FINAL_PATH):
    """
    Trains the provided neural network model.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training set.
        test_loader (DataLoader): DataLoader for the validation/test set.
        num_epochs (int): Maximum number of epochs to train.
        learning_rate (float): Initial learning rate.
        device (torch.device): The device to train on (e.g., 'cpu', 'cuda', 'mps').
        early_stopping_patience (int): Patience for early stopping.
        scheduler_factor (float): Factor for reducing learning rate.
        scheduler_patience (int): Patience for learning rate scheduler.
        weight_decay (float): Weight decay (L2 penalty) for the optimizer.
        model_save_path (str): Path to save the best model's state dictionary.
        final_model_info_path (str): Path to save final model info (including state dict).


    Returns:
        tuple: (train_losses, val_losses, best_epoch) containing training history.
               Returns ([], [], 0) if training cannot proceed.
    """
    start_time = time.time()

    # Determine device if not provided
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS acceleration for training.")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA acceleration for training.")
        else:
            device = torch.device("cpu")
            print("Using CPU for training.")
    model.to(device)

    # Ensure model directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Calculate trainable parameters
    try:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Trainable Parameters: {total_params:,}")
    except Exception as e:
        print(f"Could not calculate model parameters: {e}")

    # Loss Function (Standard MSE for pure LSTM)
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning Rate Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',           # Reduce LR when validation loss stops decreasing
        factor=scheduler_factor,
        patience=scheduler_patience,
        verbose=True,
        min_lr=1e-7           # Set a minimum learning rate
    )

    # Check if DataLoaders are valid
    try:
         if len(train_loader.dataset) == 0 or len(test_loader.dataset) == 0:
              print("Error: Training or validation dataset is empty. Cannot train model.")
              return [], [], 0
    except TypeError: # Catch potential errors if dataset doesn't have __len__ (unlikely for TensorDataset)
         print("Error: Could not determine dataset lengths. Cannot train model.")
         return [], [], 0


    # Training loop variables
    best_val_loss = float('inf')
    best_epoch = 0
    early_stopping_counter = 0
    train_losses = []
    val_losses = []

    print(f"Starting training for up to {num_epochs} epochs...")
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # --- Training Phase ---
        model.train() # Set model to training mode
        running_train_loss = 0.0
        train_batches = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()       # Clear previous gradients
            outputs = model(inputs)     # Forward pass
            loss = criterion(outputs, targets) # Calculate loss
            loss.backward()             # Backpropagation
            # Gradient Clipping (important for RNNs)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()            # Update weights

            running_train_loss += loss.item()
            train_batches += 1

        # Calculate average training loss for the epoch
        epoch_train_loss = running_train_loss / train_batches if train_batches > 0 else 0
        train_losses.append(epoch_train_loss)

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        running_val_loss = 0.0
        val_batches = 0
        epoch_val_loss = float('inf') # Default if no validation data

        if len(test_loader) > 0:
             with torch.no_grad(): # Disable gradient calculations for validation
                  for inputs, targets in test_loader:
                       inputs, targets = inputs.to(device), targets.to(device)
                       outputs = model(inputs)
                       loss = criterion(outputs, targets)
                       running_val_loss += loss.item()
                       val_batches += 1

             epoch_val_loss = running_val_loss / val_batches if val_batches > 0 else float('inf')
             val_losses.append(epoch_val_loss)
             # Update the learning rate scheduler based on validation loss
             scheduler.step(epoch_val_loss)
        else:
             # If no validation data, append NaN or skip validation loss recording
             val_losses.append(float('nan')) # Or np.nan if numpy is imported
             print(f"Warning: Validation loader is empty. Cannot compute validation loss for epoch {epoch+1}.")
             # Cannot perform early stopping or LR scheduling based on validation loss


        # --- Logging ---
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        # Print progress periodically
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {epoch_train_loss:.6f}, '
                  f'Val Loss: {epoch_val_loss:.6f}, ' # Will print inf if no validation
                  f'Time: {epoch_time:.2f}s, '
                  f'LR: {current_lr:.6e}')

        # --- Early Stopping & Best Model Saving ---
        # Only perform if validation loss is valid
        if epoch_val_loss != float('inf') and epoch_val_loss != float('nan'):
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch + 1
                try:
                    # Save only the state dictionary of the best model
                    torch.save(model.state_dict(), model_save_path)
                    early_stopping_counter = 0
                    print(f"  -> New best model saved to {model_save_path} (Epoch {best_epoch}, Val Loss: {best_val_loss:.6f})")
                except Exception as e:
                    print(f"  -> Error saving best model: {e}")
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}. Best epoch was {best_epoch}.")
                    break
        # If no validation data, cannot early stop based on it
        elif len(test_loader) == 0 and epoch > early_stopping_patience: # Alternative: stop if training loss plateaus? More complex.
             print("No validation data for early stopping. Training continues...")


    # --- Post-Training ---
    total_time = time.time() - start_time
    print(f"\nTraining finished. Total time: {total_time:.2f}s")

    # Load the best model state if one was saved
    if best_epoch > 0 and os.path.exists(model_save_path):
        print(f"Loading best model state from epoch {best_epoch}...")
        try:
            model.load_state_dict(torch.load(model_save_path))
            print(f"Best model validation loss: {best_val_loss:.6f}")

            # Save final model info (including state dict and hyperparameters)
            try:
                 # Try to get model params dynamically for saving context
                input_size = getattr(model, 'input_norm', nn.Identity()).normalized_shape[0] if isinstance(getattr(model, 'input_norm', None), nn.LayerNorm) else 'unknown'
                hidden_size = getattr(model, 'hidden_size', 'unknown')
                num_layers = getattr(model, 'num_layers', 'unknown')
                output_size = getattr(model.output_net[-1], 'out_features', 'unknown') if hasattr(model, 'output_net') and isinstance(model.output_net[-1], nn.Linear) else 'unknown'
                dense_units = getattr(model.output_net[0], 'out_features', 'unknown') if hasattr(model, 'output_net') and isinstance(model.output_net[0], nn.Linear) else 'unknown'
                dropout = 'unknown' # Harder to get reliably, might need to access specific layers

                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'best_val_loss': best_val_loss,
                    # Include hyperparams used for this model
                    'input_size': input_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'output_size': output_size,
                    'dense_units': dense_units,
                    'dropout': dropout, # Record approximate dropout used
                }, final_model_info_path)
                print(f"Final best model info saved to {final_model_info_path}")
            except Exception as e:
                 print(f"Error saving final model info: {e}")

        except Exception as e:
            print(f"Error loading best model state from {model_save_path}: {e}")
            print("Proceeding with the model state at the end of training.")
            best_epoch = 0 # Indicate that the loaded model might not be the best

    elif best_epoch == 0:
        print("No improvement found during training based on validation loss.")
        # Optionally save the final state even if it wasn't the best
        # torch.save({ 'epoch': num_epochs, 'model_state_dict': model.state_dict(), ... }, final_model_info_path)

    return train_losses, val_losses, best_epoch