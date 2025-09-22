import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, classification_report

def create_comparison_df(y_true, abnormal_users):
    
    # Create user IDs from 1 to 200 with proper formatting
    user_ids = [f"USR_{str(i).zfill(3)}" for i in range(1, 201)]
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'userID': user_ids,
        'Label': [1 if user in y_true else 0 for user in user_ids],
        'DetectedAbnormal': [1 if user in abnormal_users else 0 for user in user_ids]
    })
    return comparison_df

def evaluate_model_performance(Label, DetectedAbnormal):
    """
    Evaluate model performance using confusion matrix and F1 score
    
    Args:
        Label: True labels (ground truth)
        DetectedAbnormal: Predicted labels from the model
        
    Returns:
        f1: F1 score
        cm: Confusion matrix
    """
    # Calculate confusion matrix
    cm = confusion_matrix(Label, DetectedAbnormal)
    
    # Calculate F1 score
    f1 = f1_score(Label, DetectedAbnormal)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(Label, DetectedAbnormal))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return f1, cm