import pandas as pd
import os

from sklearn.preprocessing import StandardScaler

def load_datasets(input_folder):
    # List all parquet files in the input folder
    files = [f for f in os.listdir(input_folder) if f.endswith(".parquet")]

    if not files:
        print("Error: No parquet files found in the input folder.")
        return None

    datasets = {}
    for f in files:
        file_path = os.path.join(input_folder, f)
        try:
            df = pd.read_parquet(file_path)
            datasets[f] = df
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    if len(datasets) != 1:
        print(f"Error: Expected 1 dataset, but found {len(datasets)} parquet files.")
        return None
    return datasets


def generate_user_behavior_vectors(df, timestamp_col="Timestamp", user_col="UserID"):

    copy_df = df.copy()
    
    print(copy_df.columns.tolist())

    # Parse timestamp and extract date/hour
    copy_df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    copy_df["Date"] = copy_df[timestamp_col].dt.date
    copy_df["Hour"] = copy_df[timestamp_col].dt.hour

    # VPN / Onsite connection ratio
    def vpn_ratio(x):
        return (x == 0).mean()

    # Classify night shift hours
    copy_df["IsNight"] = copy_df["Hour"].apply(
        lambda h: 1 if (h < 7 or h >= 20 or h == 0) else (0 if 9 <= h <= 17 else None)
    )

    # Calculate shift logic: balanced ratio between night and day usage
    def calculate_shift_logic(group):
        total_logs = len(group)
        night_logs = group["IsNight"].sum()
        day_logs = total_logs - night_logs
        night_ratio = night_logs / total_logs
        day_ratio = day_logs / total_logs
        return min(night_ratio, day_ratio)

    # Group by user and date
    grouped = copy_df.groupby([user_col, "Date"])

    # Feature engineering (std_hour was removed)
    session_vectors = grouped.agg(
        total_logs=("ID", "count"),
        mean_duration=("AccessDuration", "mean"),
        fail_ratio=("IsAccessFail", "mean"),
        sensitive_ratio=("IsSensitive", "mean"),
        vpn_ratio=("Connection", vpn_ratio),
        unique_patient_count=("PatientID", pd.Series.nunique),
        unique_device_count=("DeviceID", pd.Series.nunique),
        shift_logic=("IsNight", lambda x: calculate_shift_logic(
            x.to_frame().join(copy_df.loc[x.index, ["Hour"]])
        ))
    ).fillna(0).reset_index()

    return session_vectors


def return_scaled_matrix(test):
    # Get feature names 
    feature_names = [ 'total_logs', 'mean_duration', 'fail_ratio', 'sensitive_ratio',
    'vpn_ratio', 'unique_patient_count', 'unique_device_count', 'shift_logic']
    
    # Drop identifiers before scaling
    X_test = test.drop(columns=["UserID", "Date"])
    X_test = X_test[feature_names]  # Ensure consistent column order
    
    # Use standard scaler to scale each dataset, fit only on training data
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    
    # Return DataFrame with column names
    return pd.DataFrame(X_test_scaled, columns=feature_names)


def behaviour_analysis(input_folder, output_path, user_path):
    """
    Performs the complete user behavior analysis:
    - Loads raw CSV datasets
    - Generates daily user behavior vectors
    - Scales the features
    - Saves the processed data to output path (as parquet)
    """
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(user_path, exist_ok=True)
    
    datasets = load_datasets(input_folder)
    if datasets is None:
        return
       
    # Get the first (and only) DataFrame from the datasets dictionary
    test = next(iter(datasets.values()))
    
    print("\nProcessing DataFrame with columns:", test.columns.tolist())
    df_test = generate_user_behavior_vectors(test)
    
    # Save user vectors as parquet with a dynamic filename based on input file
    input_filename = os.path.splitext(os.path.basename(next(iter(datasets.keys()))))[0]
    
    user_vectors_file = os.path.join(user_path, f"{input_filename}_raw.parquet")
    df_test.to_parquet(user_vectors_file, index=False)
    
    test_scaled = return_scaled_matrix(df_test)
    
    # Save processed data to parquet
    output_file = os.path.join(output_path, f"{input_filename}.parquet")
    test_scaled.to_parquet(output_file, index=False)
    
    print(f"\n✅ All datasets processed and saved successfully in: {output_path}")