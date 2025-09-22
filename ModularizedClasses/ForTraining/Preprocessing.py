import pandas as pd
import numpy as np
import os

def get_data(file_path, parquet_output_path=None):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Just csv or xlsx files accepted.")

    if parquet_output_path is None:
        parquet_output_path = os.path.splitext(file_path)[0] + ".parquet"
    df.to_parquet(parquet_output_path, index=False)
    print(f"Veri Parquet olarak kaydedildi: {parquet_output_path}")

    return df

def set_dataset(Train, Validation, Test):
    ext_train = os.path.splitext(Train)[1].lower()
    ext_cv = os.path.splitext(Validation)[1].lower()
    ext_test = os.path.splitext(Test)[1].lower()

    if ext_train == ".parquet":
        df_train = pd.read_parquet(Train)
    else:
        df_train = pd.read_csv(Train)

    if ext_cv == ".parquet":
        df_cv = pd.read_parquet(Validation)
    else:
        df_cv = pd.read_csv(Validation)

    if ext_test == ".parquet":
        df_test = pd.read_parquet(Test)
    else:
        df_test = pd.read_csv(Test)

    print("Datasets loaded.")
    return df_train, df_cv, df_test
    
def combine_dataset(df_train, df_cv, df_test):
    df_train = df_train.copy()
    df_cv = df_cv.copy()
    df_test = df_test.copy()
    
    df_train['source'] = 'train'
    df_cv['source'] = 'cv'
    df_test['source'] = 'test'
    
    df_all = pd.concat([df_train, df_cv, df_test], ignore_index=True)
    print("Combination Completed")
    return df_all


def convert_to_numeric(df):
    # UserID
    if "UserID" in df.columns:
        df["UserID"] = df["UserID"].str.extract(r'USR_(\d+)')
        df['UserID'] = pd.to_numeric(df['UserID'], errors='coerce').astype('Int64')
    
    # Department
    if "Department" in df.columns:
        df['Department'] = df['Department'].str.extract(r'DPT_(\d+)')
        df['Department'] = pd.to_numeric(df['Department'], errors='coerce').astype('Int64')
    
    # UserRole
    if "UserRole" in df.columns:
        df['UserRole'] = df['UserRole'].astype('category').cat.codes
    
    # Connection
    if "Connection" in df.columns:
        df["Connection"] = df["Connection"].astype('category').cat.codes
    
    # AccessLevel
    if "AccessLevel" in df.columns:
        df['AccessLevel'] = df['AccessLevel'].astype('category').cat.codes
    
    # DeviceID
    if "DeviceID" in df.columns:
        df['DeviceID'] = df['DeviceID'].str.extract(r'DVC_(\d+)')
        df['DeviceID'] = pd.to_numeric(df['DeviceID'], errors='coerce').astype('Int64')        
    
    # PatientID
    if "PatientID" in df.columns:
        df['PatientID'] = df['PatientID'].str.extract(r'DVC_(\d+)')
        df['PatientID'] = pd.to_numeric(df['PatientID'], errors='coerce').astype('Int64')
    
    # VisitDepartment
    if "VisitDepartment" in df.columns:
        df['VisitDepartment'] = df['VisitDepartment'].str.extract(r'DPT_(\d+)')
        df['VisitDepartment'] = pd.to_numeric(df['VisitDepartment'], errors='coerce').astype('Int64')
    
    # Timestamp
    if "Timestamp" in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', utc=True)
        if df['Timestamp'].isna().sum() > 0:
            print("Dikkat! Bazı timestamp değerleri parse edilemedi ve NaT oldu.")
        
        df['Year'] = df['Timestamp'].dt.year
        df['Month'] = df['Timestamp'].dt.month
        df['Day'] = df['Timestamp'].dt.day
        df['Hour'] = df['Timestamp'].dt.hour
        df['Minute'] = df['Timestamp'].dt.minute
        df['Second'] = df['Timestamp'].dt.second
        
    return df

def seperate_dataset(df_all):
    df_train = df_all[df_all['source'] == 'train'].reset_index(drop=True)
    df_cv = df_all[df_all['source'] == 'cv'].reset_index(drop=True)
    df_test = df_all[df_all['source'] == 'test'].reset_index(drop=True)
    print("Seperation Completed")
    return df_train, df_cv, df_test

def save_dataset(df, name):
    df.to_csv(name, index=False)
    
def preprocess(train_path, cv_path, test_path, output_folder="ModularizedClasses/ForTraining/outputs/"):
    """
    Runs the entire data processing pipeline with a single function call:
    1. Loads datasets from CSV or Excel files
    2. Converts all relevant columns to numeric/categorical types
    3. Combines datasets and adds a 'source' column
    4. Optionally separates and saves processed datasets as Parquet files

    Args:
        train_path (str): Path to the train dataset file (CSV or XLSX)
        cv_path (str): Path to the validation dataset file (CSV or XLSX)
        test_path (str): Path to the test dataset file (CSV or XLSX)
        output_folder (str): Folder to save the processed Parquet files (default: './outputs/')

    Returns:
        tuple: (df_train_new, df_cv_new, df_test_new, df_all)
    """
    # Step 1: Load and convert each dataset to DataFrame
    df_train = get_data(train_path)
    df_cv = get_data(cv_path)
    df_test = get_data(test_path)
    
    # Step 2: Combine all datasets and add a 'source' column
    df_all = combine_dataset(df_train, df_cv, df_test)
    
    # Step 3: Convert categorical/string columns to numeric/categorical
    df_all = convert_to_numeric(df_all)
    
    # Step 4: Separate combined dataset back into individual sets (if needed)
    df_train_new, df_cv_new, df_test_new = seperate_dataset(df_all)
    
    # Step 5: Save all datasets as Parquet files (create folder if it doesn't exist)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    df_train_new.to_parquet(os.path.join(output_folder, "train_processed.parquet"), index=False)
    df_cv_new.to_parquet(os.path.join(output_folder, "cv_processed.parquet"), index=False)
    df_test_new.to_parquet(os.path.join(output_folder, "test_processed.parquet"), index=False)

    print(f"✅ All operations completed successfully. Processed files have been saved in: {output_folder}")

    # Optionally return all processed DataFrames
    return df_train_new, df_cv_new, df_test_new, df_all