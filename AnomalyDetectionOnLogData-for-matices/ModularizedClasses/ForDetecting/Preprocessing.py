import pandas as pd
import numpy as np
import os


def get_data(file_path, csv_output_path=None):

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Just csv or xlsx files accepted.")

    if csv_output_path is None:
        csv_output_path = os.path.splitext(file_path)[0] + ".csv"
    df.to_csv(csv_output_path, index=False)
    print(f"Veri CSV olarak kaydedildi: {csv_output_path}")

    return df


def set_dataset(Test):
    df_test = pd.read_csv(Test)
    print("Dataset saved.")

    return df_test


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
            print("Warning: Some timestamp values could not be parsed and were set to NaT.")

        df['Year'] = df['Timestamp'].dt.year
        df['Month'] = df['Timestamp'].dt.month
        df['Day'] = df['Timestamp'].dt.day
        df['Hour'] = df['Timestamp'].dt.hour
        df['Minute'] = df['Timestamp'].dt.minute
        df['Second'] = df['Timestamp'].dt.second

    return df


def save_dataset(df, name):
    df.to_parquet(name, index=False)


def process_all(test_path, output_folder="ModularizedClasses/ForDetecting/outputs/"):
    """
    Runs the entire data processing pipeline with a single function call:
    1. Loads datasets from CSV or Excel files
    2. Converts all relevant columns to numeric/categorical types
    3. Combines datasets and adds a 'source' column
    4. Optionally separates and saves processed datasets as CSV files

    Args:
        test_path (str): Path to the test dataset file (CSV or XLSX)
        output_folder (str): Folder to save the processed CSV files (default: 'outputs/')

    Returns:
        dataframe: (df_test_new)
    """
    # Step 1: Load and convert each dataset to CSV format if needed
    df_test = get_data(test_path)

    # Step 2: Convert categorical/string columns to numeric/categorical
    df_test = convert_to_numeric(df_test)

    # Step 5: Save the processed dataset as a CSV file with its original name
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    base_name = os.path.splitext(os.path.basename(test_path))[0]
    save_dataset(df_test, os.path.join(output_folder, f"{base_name}_processed.parquet"))

    print("All operations completed successfully. Processed files have been saved.")

    # Optionally return all processed DataFrames
    return df_test

