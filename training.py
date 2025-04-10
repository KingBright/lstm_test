# training.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import os
import numpy as np # Import numpy for isnan/isfinite checks
import config # Import config for hyperparameters and paths
# model.py is not directly needed here as the model instance is passed in

def train_model(model, train_loader, val_loader, num_epochs=config.NUM_EPOCHS,
                learning_rate=config.LEARNING_RATE, device=None,
                early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
                scheduler_factor=config.SCHEDULER_FACTOR,
                scheduler_patience=config.SCHEDULER_PATIENCE,
                weight_decay=config.WEIGHT_DECAY,
                model_save_path=config.MODEL_BEST_PATH,
                final_model_info_path=config.MODEL_FINAL_PATH):
    """
    Trains the provided neural network model using specified data loaders and hyperparameters.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader or None): DataLoader for the validation set. Can be None.
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
            device = torch.device("mps"); print("Using MPS acceleration for training.")
        elif torch.cuda.is_available():
            device = torch.device("cuda"); print("Using CUDA acceleration for training.")
        else:
            device = torch.device("cpu"); print("Using CPU for training.")
    model.to(device)

    # Ensure model directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Calculate trainable parameters
    try:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Trainable Parameters: {total_params:,}")
    except Exception as e:
        print(f"Could not calculate model parameters: {e}")

    # Loss Function (Standard MSE suitable for state or delta prediction)
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning Rate Scheduler (ReduceLROnPlateau based on validation loss)
    # Initialize scheduler even if val_loader is None, step() will handle it
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',           # Reduce LR when validation loss stops decreasing
        factor=scheduler_factor,
        patience=scheduler_patience,
        verbose=True,
        min_lr=1e-7           # Set a minimum learning rate
    )

    # Check if DataLoaders are valid and have data
    train_loader_len = 0
    val_loader_len = 0
    try:
         train_loader_len = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else len(train_loader)
         # val_loader might be None
         val_loader_len = len(val_loader.dataset) if val_loader and hasattr(val_loader, 'dataset') else (len(val_loader) if val_loader else 0)

         if train_loader_len == 0:
              print("Error: Training dataset is empty. Cannot train model.")
              return [], [], 0
         if val_loader_len == 0:
              print("Warning: Validation dataset is empty or loader is None. Training will proceed without validation metrics, LR scheduling, or early stopping based on validation.")

    except (TypeError, AttributeError) as e:
         print(f"Error: Could not determine dataset lengths: {e}. Cannot reliably train model.")
         return [], [], 0


    # Training loop variables
    best_val_loss = float('inf')
    best_epoch = 0
    early_stopping_counter = 0
    train_losses = []
    val_losses = [] # Store NaN if no validation

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

            # Check for NaN/Inf loss during training
            if not torch.isfinite(loss):
                print(f"Error: NaN or Inf loss detected during training epoch {epoch+1}. Stopping training.")
                return train_losses, val_losses, 0 # Indicate failure

            loss.backward()             # Backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient Clipping
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
        epoch_val_loss = float('nan') # Use NaN if no validation data

        if val_loader is not None and val_loader_len > 0: # Check if val_loader exists and is not empty
             with torch.no_grad(): # Disable gradient calculations for validation
                  for inputs, targets in val_loader:
                       inputs, targets = inputs.to(device), targets.to(device)
                       outputs = model(inputs)
                       loss = criterion(outputs, targets)
                       if not torch.isfinite(loss):
                           print(f"Warning: NaN or Inf loss detected during validation epoch {epoch+1}.")
                           running_val_loss = float('nan'); break # Stop validating this epoch
                       running_val_loss += loss.item()
                       val_batches += 1

             # Calculate average validation loss only if valid numbers were summed
             if val_batches > 0 and np.isfinite(running_val_loss):
                  epoch_val_loss = running_val_loss / val_batches
             else:
                  epoch_val_loss = float('nan') # Keep NaN if validation failed or was empty

             val_losses.append(epoch_val_loss)
             # Update the learning rate scheduler based on validation loss (if valid)
             if np.isfinite(epoch_val_loss):
                  scheduler.step(epoch_val_loss)
             else:
                  print("Skipping LR scheduler step due to invalid validation loss.")

        else: # If no validation loader
             val_losses.append(float('nan')) # Record NaN for validation loss


        # --- Logging ---
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        # Print progress periodically
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            val_loss_str = f"{epoch_val_loss:.6f}" if np.isfinite(epoch_val_loss) else "N/A"
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {epoch_train_loss:.6f}, '
                  f'Val Loss: {val_loss_str}, '
                  f'Time: {epoch_time:.2f}s, '
                  f'LR: {current_lr:.6e}')

        # --- Early Stopping & Best Model Saving ---
        # Only perform if validation loss is valid
        if np.isfinite(epoch_val_loss):
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch + 1
                try:
                    torch.save(model.state_dict(), model_save_path)
                    early_stopping_counter = 0
                    # print(f"  -> New best model saved to {model_save_path} (Epoch {best_epoch}, Val Loss: {best_val_loss:.6f})") # Less verbose logging
                except Exception as e:
                    print(f"  -> Error saving best model: {e}")
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch + 1} based on validation loss. Best epoch was {best_epoch}.")
                    break
        # If no validation data, cannot early stop based on it.
        elif val_loader_len == 0 and epoch > early_stopping_patience:
             # Optional: Stop based on training loss plateau
             if epoch > early_stopping_patience + 5 and len(train_losses) > 5:
                  recent_train_losses = train_losses[-5:]
                  if all(abs(recent_train_losses[i] - recent_train_losses[i-1]) < 1e-6 for i in range(1, 5)): # Stricter plateau check
                       print(f"Stopping early at epoch {epoch+1} due to training loss plateau (no validation data).")
                       break


    # --- Post-Training ---
    total_time = time.time() - start_time
    print(f"\nTraining finished. Total time: {total_time:.2f}s")

    # Prepare final model information dictionary
    final_model_info = {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss if np.isfinite(best_val_loss) else None,
        'model_type': config.MODEL_TYPE, # Save the type that was trained
        'final_epoch_ran': epoch + 1,
        # Training hyperparameters for context
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'batch_size': config.BATCH_SIZE, # Record batch size used
        'sequence_length': config.SEQUENCE_LENGTH # Record sequence length used
    }

    # Load the best model state if one was saved, otherwise use the final state
    if best_epoch > 0 and os.path.exists(model_save_path):
        print(f"Loading best model state from epoch {best_epoch}...")
        try:
            model.load_state_dict(torch.load(model_save_path))
            print(f"Best model validation loss: {best_val_loss:.6f}")
            final_model_info['model_state_dict'] = model.state_dict() # Use best state
        except Exception as e:
            print(f"Error loading best model state from {model_save_path}: {e}")
            print("Saving the model state at the end of training instead.")
            final_model_info['model_state_dict'] = model.state_dict() # Save last state
            final_model_info['comment'] = "Best state loading failed, saved final state."
            best_epoch = 0 # Indicate best state wasn't successfully loaded/saved
    else:
        if best_epoch == 0: print("No improvement found during training based on validation loss.")
        else: print(f"Warning: Best model file {model_save_path} not found.")
        print("Saving the model state at the end of training.")
        final_model_info['model_state_dict'] = model.state_dict() # Save the final state

    # Save final model info (including state dict and hyperparameters)
    try:
        # Add model architecture parameters from config
        current_params = config.get_current_model_params()
        final_model_info.update({
            'input_size': getattr(getattr(model, 'input_norm', None), 'normalized_shape', ['unknown'])[0] if hasattr(model, 'input_norm') else 'unknown',
            'hidden_size': current_params.get('hidden_size', 'unknown'),
            'num_layers': current_params.get('num_layers', 'unknown'),
            'output_size': getattr(getattr(model, 'output_net', nn.Sequential())[-1], 'out_features', 'unknown') if hasattr(model, 'output_net') and len(getattr(model, 'output_net', nn.Sequential())) > 0 and isinstance(getattr(model, 'output_net', nn.Sequential())[-1], nn.Linear) else 'unknown',
            'dense_units': current_params.get('dense_units', 'unknown'),
            'dropout_rate': current_params.get('dropout_rate', 'unknown')
        })
        torch.save(final_model_info, final_model_info_path)
        print(f"Final model info (including state dict) saved to {final_model_info_path}")
    except Exception as e:
         print(f"Error saving final model info: {e}")

    # Return training history and best epoch number
    return train_losses, val_losses, best_epoch

