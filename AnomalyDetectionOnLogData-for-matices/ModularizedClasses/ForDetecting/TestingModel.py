import os
import pandas as pd
import numpy as np

def print_results_generalized(total, normal, anomalies, reconstruction_error):
    print("\n" + "="*50)
    print("🔍 ANOMALY DETECTION RESULTS BY LSTM AUTOENCODER")
    print("="*50)
    print(f"📊 Total Sequences: {total}")
    print(f"✅ Normal: {normal} ({(normal/total)*100:.2f}%)")
    print(f"⚠️ Anomalies: {anomalies} ({(anomalies/total)*100:.2f}%)")
    print(f"📈 Average Error: {np.mean(reconstruction_error):.6f}")
    print("="*50 + "\n")

def setDetectedAbnormalUsers(df, predictions, label="Test"):
    abnormal_users = []
    anomalous_indices = np.where(predictions == 1)[0]
    
    if len(anomalous_indices) == 0:
        return abnormal_users
    
    for idx in anomalous_indices:
        row = df.iloc[idx]
        user = row['UserID'] if 'UserID' in row else 'Unknown'
        if user != 'Unknown':
            # UserID'yi formatlama
            user_num = int(user)  # String'i integer'a çevir
            if user_num < 10:  # Tek basamaklı
                formatted_user = f"USR_00{user_num}"
            elif user_num < 100:  # İki basamaklı
                formatted_user = f"USR_0{user_num}"
            else:  # Üç basamaklı
                formatted_user = f"USR_{user_num}"
            
            abnormal_users.append(formatted_user)
    
    return list(set(abnormal_users))

def print_abnormal_behaviour_rows(df, predictions, label="Test"):
    # find abnormals
    anomalous_indices = np.where(predictions == 1)[0]
    if len(anomalous_indices) == 0:
        print(f"[{label}] Hiç anomali tespit edilmedi.")
        return
    
    for idx in anomalous_indices:
        row = df.iloc[idx]
        user = row['UserID'] if 'UserID' in row else 'Unknown'
        date = row['Date'] if 'Date' in row else 'Unknown'
        print(f"[{label}] User {user} on {date} marked as anomaly.")
    
def DetectAbnormalBehaviour(model_predictor, threshold_num, data_path, raw_df_path):
    """
    Detect abnormal behaviors using either LSTM or standard autoencoder
    """
    # Load the model
    autoencod = model_predictor
    print(f"Model loaded. Expected input shape: {autoencod.input_shape}")
    
    # Load data
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
        raw_df = pd.read_parquet(raw_df_path)
    else:
        df = pd.read_csv(data_path)
        raw_df = pd.read_csv(raw_df_path)
    
    # Select features
    expected_features = ['total_logs', 'mean_duration', 'fail_ratio', 'sensitive_ratio',
                        'vpn_ratio', 'unique_patient_count', 'unique_device_count', 'shift_logic']
    
    if not all(feature in df.columns for feature in expected_features):
        raise ValueError(f"Missing features. Required: {expected_features}")
    
    # Prepare data
    data = df[expected_features].values.astype('float32')
    print(f"Original data shape: {data.shape}")
    
    # Check model type and reshape accordingly
    input_shape = autoencod.input_shape
    if len(input_shape) == 3:  # LSTM model expects (samples, timesteps, features)
        print("LSTM model detected - reshaping to 3D")
        data = np.reshape(data, (data.shape[0], 1, data.shape[1]))
    else:  # Standard autoencoder expects (samples, features)
        print("Standard autoencoder detected - using 2D shape")
    
    print(f"Input data shape after reshape: {data.shape}")
    
    # Predict
    reconstructed = autoencod.predict(data, verbose=0)
    
    # Reshape back if needed
    if len(input_shape) == 3:
        reconstructed = np.reshape(reconstructed, (reconstructed.shape[0], reconstructed.shape[2]))
        data = np.reshape(data, (data.shape[0], data.shape[2]))
    
    # Calculate reconstruction error
    reconstruction_error = np.mean(np.square(data - reconstructed), axis=1)
    
    # Identify abnormal behaviours    
    test_pred = (reconstruction_error > threshold_num).astype(int)
    
    # Print results
    total = len(reconstruction_error)
    anomalies = np.sum(test_pred)
    normal = total - anomalies
    
    abnormal_users = setDetectedAbnormalUsers(raw_df, test_pred, label="Test")
    print_results_generalized(total, normal, anomalies, reconstruction_error)
    print_abnormal_behaviour_rows(raw_df, test_pred, label="Test")
    
    return test_pred, reconstruction_error, abnormal_users