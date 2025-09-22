import os
import sys
import pandas as pd
from Preprocessing import process_all
from BehaviourAnalysis import behaviour_analysis
from TestingModel import DetectAbnormalBehaviour
from keras.models import load_model
from Evaluation import create_comparison_df, evaluate_model_performance
#from SecondPhase import explain_anomalies
EXPL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Explainability'))
if EXPL_DIR not in sys.path:
    sys.path.insert(0, EXPL_DIR)
try:
    from explainability import (
        global_summary_from_behaviours,
        global_ig_summary_from_behaviours,
        anomaly_report,
    )
except Exception:
    global_summary_from_behaviours = None
    global_ig_summary_from_behaviours = None
    anomaly_report = None

input = r"DatasetGenerator\GeneratingSyntheticLogDatas\TrdTry\Test\Test.csv"
Model = r"ModularizedClasses\Model\autoencoder_model.keras"
Label = r"DatasetGenerator\GeneratingSyntheticLogDatas\TrdTry\Test\Test_AnomalousUsers.txt"

outputPath = r"ModularizedClasses\ForDetecting\outputs"
behaviourPath = r"ModularizedClasses\ForDetecting\behaviours"
userPath = r"ModularizedClasses\ForDetecting\users"

Threshold = 0.452005

if __name__ == "__main__":
    # Resolve absolute paths relative to project root to avoid CWD issues
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    input = os.path.join(
        PROJECT_ROOT,
        'DatasetGenerator', 'GeneratingSyntheticLogDatas', 'TrdTry', 'Test', 'Test.csv'
    )
    Model = os.path.join(PROJECT_ROOT, 'ModularizedClasses', 'Model', 'autoencoder_model.keras')
    Label = os.path.join(
        PROJECT_ROOT,
        'DatasetGenerator', 'GeneratingSyntheticLogDatas', 'TrdTry', 'Test', 'Test_AnomalousUsers.txt'
    )
    outputPath = os.path.join(PROJECT_ROOT, 'ModularizedClasses', 'ForDetecting', 'outputs')
    behaviourPath = os.path.join(PROJECT_ROOT, 'ModularizedClasses', 'ForDetecting', 'behaviours')
    userPath = os.path.join(PROJECT_ROOT, 'ModularizedClasses', 'ForDetecting', 'users')
    process_all(input, outputPath)
    behaviour_analysis(outputPath, behaviourPath, userPath)
    
    # Load the model
    autoencod = load_model(Model)
    
    predictions, errors, abnormal_users = DetectAbnormalBehaviour(
        model_predictor = autoencod,
        threshold_num=Threshold,
        data_path= behaviourPath + r"\Test_processed.parquet",
        raw_df_path= userPath + r"\Test_processed_raw.parquet"
    )

    # Optional explainability artifacts on test set
    try:
        if global_summary_from_behaviours is not None:
            exp_out = os.path.join(behaviourPath, 'explainability')
            global_summary_from_behaviours(behaviourPath, 'test_processed', autoencod, out_dir=exp_out)
        if global_ig_summary_from_behaviours is not None:
            exp_out = os.path.join(behaviourPath, 'explainability')
            global_ig_summary_from_behaviours(behaviourPath, 'test_processed', autoencod, out_dir=exp_out)
        if anomaly_report is not None:
            behaviours_file = os.path.join(behaviourPath, 'Test_processed.parquet')
            users_file = os.path.join(userPath, 'Test_processed_raw.parquet')
            _ = anomaly_report(autoencod, behaviours_file, users_file, threshold=Threshold)
    except Exception as e:
        print(f"Explainability (test) skipped: {e}")
    
    # Get true labels from file
    # Remove duplicates from abnormal_users
    abnormal_users = list(set(abnormal_users))
    # Debug print
    print("\nDetected Abnormal Users:")
    print(f"Total count: {len(abnormal_users)}")
    for i, user in enumerate(abnormal_users, 1):
        print(f"{i}. {user}")
    y_true = []
    with open(Label, 'r') as f:
        y_true = [line.strip() for line in f]
        
    comparison_df = create_comparison_df(y_true, abnormal_users)
    evaluate_model_performance( comparison_df['Label'], comparison_df['DetectedAbnormal'])
    
