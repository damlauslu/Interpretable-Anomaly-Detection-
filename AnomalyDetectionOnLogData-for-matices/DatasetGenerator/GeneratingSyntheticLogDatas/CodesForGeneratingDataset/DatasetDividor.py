import pandas as pd
import random

def datasetDividor(input_file, n, output_selected, output_remaining):
    # Read the dataset
    df = pd.read_csv(input_file)
    
    # Ensure n is not greater than the total number of records
    if n > len(df):
        raise ValueError("n is larger than the dataset size!")
    
    # Randomly select n rows
    selected_indices = random.sample(range(len(df)), n)
    selected_rows = df.iloc[selected_indices]
    remaining_rows = df.drop(selected_indices)
    
    # Save to new CSV files
    selected_rows.to_csv(output_selected, index=False)
    remaining_rows.to_csv(output_remaining, index=False)
    
    print(f"Extracted {n} lines to {output_selected} and saved remaining lines to {output_remaining}.")

# Example usage
datasetDividor(r"./DatasetGenerator/GeneratingSyntheticLogDatas/Fourth/hospital_access_logs.csv", 400000, r"./DatasetGenerator/GeneratingSyntheticLogDatas/Fourth/train/Train.csv", r"./DatasetGenerator/GeneratingSyntheticLogDatas/Fourth/cv/CV.csv")