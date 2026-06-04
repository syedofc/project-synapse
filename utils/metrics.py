# ==============================================================================
# File 08b: 08_utils/08b_metrics.py
# Purpose: A centralized location for metric calculation functions.
# ==============================================================================
"""
A utility for calculating performance metrics.

This module provides standalone functions to compute common metrics like accuracy.
Centralizing these functions makes the training code cleaner and makes it easy
to add more complex metrics (e.g., F1-score, Precision, Recall) in the future.
"""
import torch

def calculate_accuracy(outputs, labels):
    """
    Calculates the accuracy for a batch of predictions.

    Args:
        outputs (torch.Tensor): The raw output logits from the model.
                                Shape: (batch_size, num_classes)
        labels (torch.Tensor): The ground truth labels.
                               Shape: (batch_size)

    Returns:
        float: The accuracy for the batch (0.0 to 1.0).
    """
    if outputs.size(0) == 0:
        return 0.0

    # Get the index of the max log-probability
    _, predicted = torch.max(outputs.data, 1)
    
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    
    return correct / total

# In the future, we could add more complex metrics here, for example:
# from sklearn.metrics import f1_score, precision_score, recall_score
#
# def calculate_f1(outputs, labels):
#     _, predicted = torch.max(outputs.data, 1)
#     # Ensure tensors are on CPU and are numpy arrays for sklearn
#     y_true = labels.cpu().numpy()
#     y_pred = predicted.cpu().numpy()
#     return f1_score(y_true, y_pred, average='weighted')
