import pandas as pd
import os

from sklearn.preprocessing import StandardScaler

def load_datasets(input_folder):
    """
    Load train, cv, and test datasets from the input folder (parquet format).
    Returns tuple of (train, cv, test) DataFrames or None if error occurs.
    """
    try:
        # Get all Parquet files
        parquet_files = [f for f in os.listdir(input_folder) if f.endswith('.parquet')]
        
        if len(parquet_files) != 3:
            print(f"❌ Error: Expected 3 Parquet files, found {len(parquet_files)}")
            return None
            
        # Initialize datasets
        train = cv = test = None
        
        # Read and identify datasets
        for file in parquet_files:
            df = pd.read_parquet(os.path.join(input_folder, file))
            
            # Check if file has required column
            if 'source' not in df.columns:
                print(f"❌ Error: 'source' column missing in {file}")
                continue
                
            # Identify dataset type
            source = df['source'].iloc[0].lower()
            if source == 'train':
                train = df
            elif source == 'cv':
                cv = df
            elif source == 'test':
                test = df
                
        # Verify all datasets were loaded
        if train is None or cv is None or test is None:
            print("❌ Error: Could not find all required datasets (train, cv, test)")
            return None
            
        print("✅ All datasets loaded successfully")
        return train, cv, test
        
    except Exception as e:
        print(f"❌ Error loading datasets: {str(e)}")
        return None

def generate_user_behavior_vectors(df, timestamp_col="Timestamp", user_col="UserID"):
    copy_df = df.copy()
    copy_df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    copy_df["Date"] = copy_df[timestamp_col].dt.date
    copy_df["Hour"] = copy_df[timestamp_col].dt.hour

    def vpn_ratio(x):
        return (x == 0).mean()

    copy_df["IsNight"] = copy_df["Hour"].apply(
        lambda h: 1 if (h < 7 or h >= 20 or h == 0) else (0 if 9 <= h <= 17 else None)
    )

    def calculate_shift_logic(group):
        total_logs = len(group)
        night_logs = group["IsNight"].sum()
        day_logs = total_logs - night_logs
        night_ratio = night_logs / total_logs
        day_ratio = day_logs / total_logs
        return min(night_ratio, day_ratio)

    grouped = copy_df.groupby([user_col, "Date"])

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

def return_scaled_matrix(train, cv, test):
    feature_names = ['total_logs', 'mean_duration', 'fail_ratio', 'sensitive_ratio',
                    'vpn_ratio', 'unique_patient_count', 'unique_device_count', 'shift_logic']
    
    X_train = train.drop(columns=["UserID", "Date"])[feature_names]
    X_val = cv.drop(columns=["UserID", "Date"])[feature_names]
    X_test = test.drop(columns=["UserID", "Date"])[feature_names]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_cv_scaled = scaler.fit_transform(X_val)
    X_test_scaled = scaler.fit_transform(X_test)
    
    return (pd.DataFrame(X_train_scaled, columns=feature_names),
            pd.DataFrame(X_cv_scaled, columns=feature_names),
            pd.DataFrame(X_test_scaled, columns=feature_names))

def behaviour_analysis(input_folder, output_path):
    """
    Performs the complete user behavior analysis:
    - Loads raw Parquet datasets
    - Generates daily user behavior vectors
    - Scales the features
    - Saves the processed data to output path (parquet)
    """
    os.makedirs(output_path, exist_ok=True)
    
    datasets = load_datasets(input_folder)
    if datasets is None:
        print("❌ Error: Failed to load datasets")
        return
        
    train, cv, test = datasets
    
    df_train = generate_user_behavior_vectors(train)
    df_cv = generate_user_behavior_vectors(cv)
    df_test = generate_user_behavior_vectors(test)
    
    df_train_scaled, df_cv_scaled, df_test_scaled = return_scaled_matrix(df_train, df_cv, df_test)
    
    df_train_scaled['source'] = 'train'
    df_cv_scaled['source'] = 'cv'
    df_test_scaled['source'] = 'test'
    
    # Save processed data to Parquet
    df_train_scaled.to_parquet(os.path.join(output_path, "train.parquet"), index=False)
    df_cv_scaled.to_parquet(os.path.join(output_path, "cv.parquet"), index=False)
    df_test_scaled.to_parquet(os.path.join(output_path, "test.parquet"), index=False)

    print(f"All datasets processed and saved successfully in: {output_path}")
