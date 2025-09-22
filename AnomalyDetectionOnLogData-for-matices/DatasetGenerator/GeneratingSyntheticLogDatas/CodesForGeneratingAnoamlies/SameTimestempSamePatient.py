import pandas as pd
import numpy as np
import os

# Dosya yolları
input_path = r"DatasetGenerator\GeneratingSyntheticLogDatas\Fourth\test\Test.csv"
output_path = r"DatasetGenerator\GeneratingSyntheticLogDatas\Fourth\test\Test.csv"
anomalies_path = r"DatasetGenerator\GeneratingSyntheticLogDatas\Fourth\test\Tes_anomalous_same_timestamp_same_patient.csv"
user_tracking_path = r"DatasetGenerator\GeneratingSyntheticLogDatas\Fourth\test\Test_abnormal_users.txt"

# Veriyi yükle
df = pd.read_csv(input_path)

# Daha önce anomalilere dahil edilen kullanıcıları oku
if os.path.exists(user_tracking_path):
    with open(user_tracking_path, "r") as f:
        used_user_ids = set(f.read().splitlines())
else:
    used_user_ids = set()

# En az 10 logu olan ve daha önce kullanılmamış kullanıcıları seç
eligible_users = (
    df["UserID"]
    .value_counts()
    .loc[lambda x: x > 10]
    .index
    .difference(list(used_user_ids))
    .tolist()
)

if len(eligible_users) < 2:
    raise ValueError("Anomali için yeterli sayıda yeni kullanıcı yok.")

# 3 kullanıcı seç
selected_users = np.random.choice(eligible_users, 2, replace=False)

# Ortak timestamp ve ortak patient seçimi için rastgele bir timestamp ve patient seç
reference_row = df.sample(1).iloc[0]
common_timestamp = reference_row["Timestamp"]
common_patient = reference_row["PatientID"]

# Değiştirilecek index'leri burada toplayacağız
anomalous_indices = []

for user in selected_users:
    user_logs = df[df["UserID"] == user]
    n_anomalies = int(len(user_logs) * 0.18)
    sampled = user_logs.sample(n=n_anomalies, random_state=42)
    indices = sampled.index

    # Güncelleme
    df.loc[indices, "Timestamp"] = common_timestamp
    df.loc[indices, "PatientID"] = common_patient

    anomalous_indices.extend(indices)

# Anomalileri ayrı kaydet
anomalies_df = df.loc[anomalous_indices]
anomalies_df.to_csv(anomalies_path, mode='a', index=False, header=not os.path.exists(anomalies_path))

# Güncellenmiş veri dosyasını kaydet
df.to_csv(output_path, index=False)

# Anomali uygulanan satırların ID’lerini yazdır
print("Aynı Timestamp & Aynı Hasta ID Anomali uygulanan satırların ID'leri:")
print(df.loc[anomalous_indices, "ID"].tolist())

# --- Yeni kullanıcıları user_id kayıt dosyasına ekle ---
if os.path.exists(user_tracking_path):
    with open(user_tracking_path, "r") as f:
        existing_users = set(f.read().splitlines())
else:
    existing_users = set()

updated_users = existing_users.union(map(str, selected_users))

with open(user_tracking_path, "w") as f:
    for uid in sorted(updated_users):
        f.write(uid + "\n")
