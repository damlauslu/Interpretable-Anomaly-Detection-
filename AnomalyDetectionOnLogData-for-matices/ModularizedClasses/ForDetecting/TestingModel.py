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

def print_anomaly_reasons(raw_df, X_in_2d, X_recon_2d, predictions, feature_names, label="Test", top_k=3):
    """
    For each predicted anomaly, print top-k features contributing to the
    reconstruction error with their percent contributions and input/recon values.
    """
    anomalies = np.where(predictions == 1)[0]
    if len(anomalies) == 0:
        return
    # squared errors per feature
    se = np.square(X_in_2d - X_recon_2d)  # (n, f)
    for idx in anomalies:
        row = raw_df.iloc[idx] if idx < len(raw_df) else None
        user = row.get('UserID', 'Unknown') if row is not None else 'Unknown'
        date = row.get('Date', 'Unknown') if row is not None else 'Unknown'
        errs = se[idx]
        total = errs.sum() + 1e-12
        order = np.argsort(-errs)
        parts = []
        for rnk in range(min(top_k, len(feature_names))):
            j = order[rnk]
            fname = feature_names[j]
            pct = errs[j] / total
            parts.append(
                f"{fname} {pct:.2%} (in={X_in_2d[idx, j]:.3f}, recon={X_recon_2d[idx, j]:.3f})"
            )
        reasons = "; ".join(parts)
        print(f"[{label}] Why anomaly for User {user} on {date}: {reasons}")

def _feature_phrase_map():
    return {
        'total_logs': 'unusual volume of activity',
        'mean_duration': 'atypical session duration',
        'fail_ratio': 'unusually high failure ratio',
        'sensitive_ratio': 'excessive access to sensitive data',
        'vpn_ratio': 'unusual VPN usage ratio',
        'unique_patient_count': 'unusual number of distinct patients accessed',
        'unique_device_count': 'unusual number of distinct devices used',
        'shift_logic': 'activity outside expected shift patterns',
    }

def _severity_word(pct: float) -> str:
    if pct >= 0.50:
        return 'primary'
    if pct >= 0.30:
        return 'strong'
    if pct >= 0.15:
        return 'moderate'
    return 'minor'

def print_human_explanations(raw_df, X_in_2d, X_recon_2d, predictions, feature_names, label="Test", top_k=3):
    """
    Print human-interpretable English explanations for each anomaly using
    short phrases + severity + direction (higher/lower than expected).
    """
    anomalies = np.where(predictions == 1)[0]
    if len(anomalies) == 0:
        return
    phrases = _feature_phrase_map()
    se = np.square(X_in_2d - X_recon_2d)
    for idx in anomalies:
        row = raw_df.iloc[idx] if idx < len(raw_df) else None
        user = row.get('UserID', 'Unknown') if row is not None else 'Unknown'
        date = row.get('Date', 'Unknown') if row is not None else 'Unknown'
        errs = se[idx]
        total = errs.sum() + 1e-12
        order = np.argsort(-errs)
        parts = []
        for rnk in range(min(top_k, len(feature_names))):
            j = order[rnk]
            fname = feature_names[j]
            pct = float(errs[j] / total)
            sev = _severity_word(pct)
            direction = 'higher than expected' if (X_in_2d[idx, j] - X_recon_2d[idx, j]) > 0 else 'lower than expected'
            base = phrases.get(fname, f"deviation in {fname}")
            parts.append(f"{base} ({sev}, {pct:.0%}, {direction})")
        line = "; ".join(parts)
        print(f"[{label}] Explanation for User {user} on {date}: {line}")
    
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
    # Suppress verbose prints for a single-line-per-anomaly output
    # print_results_generalized(total, normal, anomalies, reconstruction_error)
    # print_abnormal_behaviour_rows(raw_df, test_pred, label="Test")

    # Build one-line messages and also save a table to CSV/XLSX
    expected_features = ['total_logs', 'mean_duration', 'fail_ratio', 'sensitive_ratio',
                        'vpn_ratio', 'unique_patient_count', 'unique_device_count', 'shift_logic']
    anomalies_idx = np.where(test_pred == 1)[0]
    se = np.square((data if len(autoencod.input_shape) != 3 else np.reshape(data, (data.shape[0], data.shape[2]))) - reconstructed)

    rows = []
    phrases = _feature_phrase_map()
    def _severity_word(pct: float) -> str:
        if pct >= 0.50:
            return 'primary'
        if pct >= 0.30:
            return 'strong'
        if pct >= 0.15:
            return 'moderate'
        return 'minor'

    for idx in anomalies_idx:
        row = raw_df.iloc[idx] if idx < len(raw_df) else None
        user = row.get('UserID', 'Unknown') if row is not None else 'Unknown'
        date = row.get('Date', 'Unknown') if row is not None else 'Unknown'
        errs = se[idx]
        total_err = errs.sum() + 1e-12
        order = np.argsort(-errs)
        # Compose explanation (phrases + severity + direction) and measurements
        parts_expl = []
        parts_meas = []
        for rnk in range(min(3, len(expected_features))):
            j = order[rnk]
            fname = expected_features[j]
            pct = float(errs[j] / total_err)
            sev = _severity_word(pct)
            direction = 'higher than expected' if ((data if len(autoencod.input_shape) != 3 else np.reshape(data, (data.shape[0], data.shape[2])))[idx, j] - reconstructed[idx, j]) > 0 else 'lower than expected'
            base = phrases.get(fname, f"deviation in {fname}")
            parts_expl.append(f"{base} ({sev}, {pct:.0%}, {direction})")
            parts_meas.append(f"{fname} {pct:.2%} (in={((data if len(autoencod.input_shape) != 3 else np.reshape(data, (data.shape[0], data.shape[2])))[idx, j]):.3f} -> {reconstructed[idx, j]:.3f})")
        expl = "; ".join(parts_expl)
        meas = "; ".join(parts_meas)
        print(f"Id={idx}, UserID={user}, Date={date} marked as anomaly: {expl} ({meas})")
        rows.append({
            'idx': int(idx),
            'UserID': user,
            'Date': date,
            'total_error': float(reconstruction_error[idx]),
            'Explanation': expl,
            'Measurements': meas,
        })

    # Save CSV/XLSX
    try:
        if len(rows) > 0:
            table_df = pd.DataFrame(rows).sort_values('total_error', ascending=False)
            beh_dir = os.path.dirname(data_path)
            out_dir = os.path.join(beh_dir, "explainability")
            os.makedirs(out_dir, exist_ok=True)
            out_csv = os.path.join(out_dir, "anomaly_explanations_human.csv")
            table_df.to_csv(out_csv, index=False)
            try:
                out_xlsx = os.path.join(out_dir, "anomaly_explanations_human.xlsx")
                table_df.to_excel(out_xlsx, index=False)
            except Exception:
                pass
    except Exception:
        pass

    return test_pred, reconstruction_error, abnormal_users
