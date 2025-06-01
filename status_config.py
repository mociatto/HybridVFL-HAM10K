# status_config.py
# Dashboard Status Messages Configuration
# Modify these messages to customize what appears in the "CURRENT STATUS" box

STATUS_MESSAGES = {
    # Initial loading phase
    "LOADING_DATASET": "Loading HAM10K Dataset...",
    
    # Training phases
    "TRAINING_START": "Initializing {mode} Training...",
    "TRAINING_ROUND": "Training {mode} - Round {round}/{total_rounds}",
    "TRAINING_EPOCH": "Training {mode} - Round {round} Epoch {epoch}/{total_epochs}",
    "ROUND_COMPLETED": "Completed {mode} - Round {round}/{total_rounds}",
    
    # Evaluation phase
    "EVALUATING": "Evaluating {mode} Model...",
    "CALCULATING_METRICS": "Calculating Performance Metrics...",
    "CALCULATING_LEAKAGE": "Analyzing Privacy Leakage...",
    
    # Completion
    "TRAINING_COMPLETED": "Training Completed Successfully",
    "EVALUATION_COMPLETED": "Evaluation Completed",
    
    # Error states (optional)
    "ERROR_LOADING": "Error: Failed to Load Dataset",
    "ERROR_TRAINING": "Error: Training Failed",
    "ERROR_EVALUATION": "Error: Evaluation Failed",
    
    # Additional custom states you can use
    "PREPARING_DATA": "Preparing Training Data...",
    "SAVING_MODEL": "Saving Model Checkpoints...",
    "LOADING_CHECKPOINT": "Loading Model Checkpoint...",
    "OPTIMIZING_HYPERPARAMS": "Optimizing Hyperparameters...",
    "FEDERATION_SYNC": "Synchronizing Federation Clients...",
    "ADVERSARIAL_TRAINING": "Training Adversarial Components...",
    "FAIRNESS_EVALUATION": "Evaluating Fairness Constraints...",
}

# Status message helper functions
def get_status(status_key, **kwargs):
    """Get a formatted status message"""
    
    if status_key not in STATUS_MESSAGES:
        return f"Unknown Status: {status_key}"
    
    try:
        return STATUS_MESSAGES[status_key].format(**kwargs)
    except KeyError as e:
        return f"Status Error: Missing variable {e}"

# Pre-defined status combinations for common scenarios
def get_training_status(mode, round_num, total_rounds, epoch=None, total_epochs=None):
    """Get training status with automatic message selection"""
    if epoch is not None:
        return get_status("TRAINING_EPOCH", 
                         mode=mode, round=round_num, epoch=epoch, total_epochs=total_epochs)
    else:
        return get_status("TRAINING_ROUND", 
                         mode=mode, round=round_num, total_rounds=total_rounds)

def get_completion_status(mode, round_num, total_rounds):
    """Get round completion status"""
    return get_status("ROUND_COMPLETED", 
                     mode=mode, round=round_num, total_rounds=total_rounds)

def get_evaluation_status(mode):
    """Get evaluation status"""
    return get_status("EVALUATING", mode=mode)
