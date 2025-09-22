import pandas as pd
import numpy as np
import os

# Dosya yolları
input_path = r"DatasetGenerator\GeneratingSyntheticLogDatas\Fourth\test\Test.csv"
output_path = r"DatasetGenerator\GeneratingSyntheticLogDatas\Fourth\test\Test.csv"
anomalies_path = r"DatasetGenerator\GeneratingSyntheticLogDatas\Fourth\test\Test_anomalous_duplicate_timestamp.csv"
user_tracking_path = r"DatasetGenerator\GeneratingSyntheticLogDatas\Fourth\test\Test_abnormal_users.txt"

# Veriyi yükle
df = pd.read_csv(input_path)

# Daha önce anomalilere dahil edilen kullanıcıları oku
if os.path.exists(user_tracking_path):
    with open(user_tracking_path, "r") as f:
        used_user_ids = set(f.read().splitlines())
else:
    used_user_ids = set()

# Yeterli sayıda logu olan ve daha önce kullanılmamış kullanıcıları bul
eligible_users = (
    df["UserID"]
    .value_counts()
    .loc[lambda x: x > 10]  # En az 10 logu olan kullanıcılar
    .index
    .difference(list(used_user_ids))
    .tolist()
)

# Kullanıcı seçimi
if not eligible_users:
    raise ValueError("Yeni anomali uygulanabilecek kullanıcı bulunamadı.")
selected_user = np.random.choice(eligible_users)

# Kullanıcının loglarını al
user_logs = df[df["UserID"] == selected_user]
n_anomalies = int(len(user_logs) * 0.7)
anomaly_rows = user_logs.sample(n=n_anomalies, random_state=42)
indices = anomaly_rows.index

# Ortak timestamp oluştur
common_timestamp = anomaly_rows["Timestamp"].iloc[0]

# Timestampleri aynı yap
df.loc[indices, "Timestamp"] = common_timestamp

# Anomalileri kaydet
anomalies_df = df.loc[indices]
anomalies_df.to_csv(anomalies_path, mode='a', index=False, header=not os.path.exists(anomalies_path))

# Tüm veri güncellemesi
df.to_csv(output_path, index=False)

# Uygulanan ID'leri yazdır
print("Duplicate Timestamp Anomali uygulanan satırların ID'leri:")
ids = df.loc[indices, "ID"]
# Ensure ids is always printed as a list, regardless of its type
if isinstance(ids, (pd.Series, np.ndarray, list)):
    print(list(ids))
else:
    print([ids])
# --- UserID'yi txt dosyasına ekle ---
if os.path.exists(user_tracking_path):
    with open(user_tracking_path, "r") as f:
        existing_users = set(f.read().splitlines())
else:
    existing_users = set()

updated_users = existing_users.union({str(selected_user)})

with open(user_tracking_path, "w") as f:
    for uid in sorted(updated_users):
        f.write(uid + "\n")
