import os
import sys
import time

from Preprocessing import preprocess
from BehaviourAnalysis import behaviour_analysis
from TrainingModel import TrainModel, plot_training_history

# Optional explainability utilities
EXPL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Explainability'))
if EXPL_DIR not in sys.path:
    sys.path.insert(0, EXPL_DIR)
try:
    from explainability import global_summary_from_behaviours, global_ig_summary_from_behaviours
except Exception:
    global_summary_from_behaviours = None
    global_ig_summary_from_behaviours = None


trainPath = r"DatasetGenerator\GeneratingSyntheticLogDatas\TrdTry\Final\hospital_access_logs.csv"
cvPath = r"DatasetGenerator\GeneratingSyntheticLogDatas\TrdTry\CV\CV.csv"
testPath = r"DatasetGenerator\GeneratingSyntheticLogDatas\TrdTry\Test\Test.csv"

OutputPath = "ModularizedClasses/ForTraining/outputs/"
BehvaiourPath = "ModularizedClasses/ForTraining/behaviours/"
ModelPath = "ModularizedClasses/Model/"

# Make paths absolute relative to project root so execution works from any CWD
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
trainPath = os.path.join(
    PROJECT_ROOT,
    'DatasetGenerator', 'GeneratingSyntheticLogDatas', 'TrdTry', 'Final', 'hospital_access_logs.csv'
)
cvPath = os.path.join(
    PROJECT_ROOT,
    'DatasetGenerator', 'GeneratingSyntheticLogDatas', 'TrdTry', 'CV', 'CV.csv'
)
testPath = os.path.join(
    PROJECT_ROOT,
    'DatasetGenerator', 'GeneratingSyntheticLogDatas', 'TrdTry', 'Test', 'Test.csv'
)
OutputPath = os.path.join(PROJECT_ROOT, 'ModularizedClasses', 'ForTraining', 'outputs')
BehvaiourPath = os.path.join(PROJECT_ROOT, 'ModularizedClasses', 'ForTraining', 'behaviours')
ModelPath = os.path.join(PROJECT_ROOT, 'ModularizedClasses', 'Model')

if __name__ == "__main__":
    print("\nPerformance Measurement Started")
    print("=" * 50)

    start_time = time.time()

    preprocess(trainPath, cvPath, testPath, OutputPath)
    behaviour_analysis(OutputPath, BehvaiourPath)
    model, history = TrainModel(ModelPath, BehvaiourPath)

    plot_training_history(history)

    # Generate explainability summaries (global) if available
    try:
        if global_summary_from_behaviours is not None:
            exp_out = os.path.join(BehvaiourPath, 'explainability')
            global_summary_from_behaviours(BehvaiourPath, 'train', model, out_dir=exp_out)
            global_summary_from_behaviours(BehvaiourPath, 'cv', model, out_dir=exp_out)
        if global_ig_summary_from_behaviours is not None:
            exp_out = os.path.join(BehvaiourPath, 'explainability')
            global_ig_summary_from_behaviours(BehvaiourPath, 'train', model, out_dir=exp_out)
            global_ig_summary_from_behaviours(BehvaiourPath, 'cv', model, out_dir=exp_out)
    except Exception as e:
        print(f"Explainability generation skipped: {e}")

    end_time = time.time()
    total_time = end_time - start_time

    print("\nPerformance Results:")
    print("=" * 50)
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"                     {total_time/60:.2f} minutes")
    print("=" * 50)
