import pandas as pd
import numpy as np
import os

# Ayarlanabilir dosya yolları
input_path = r"DatasetGenerator\GeneratingSyntheticLogDatas\Fourth\test\Test.csv"
output_path = r"DatasetGenerator\GeneratingSyntheticLogDatas\Fourth\test\Test.csv"
anomalies_path = r"DatasetGenerator\GeneratingSyntheticLogDatas\Fourth\test\Test_anomalous_access_duration.csv"

# Veriyi yükle
df = pd.read_csv(input_path)

# Normal erişim süresi aralığına sahip kullanıcıları filtrele (120-300 sn)
valid_users = (
    df[(df["AccessDuration"] >= 120) & (df["AccessDuration"] <= 300)]
    ["UserID"]
    .value_counts()
    .loc[lambda x: x > 5]  # En az 5 logu olan kullanıcılar
    .index
    .tolist()
)

# Rastgele 3 kullanıcı seç
selected_users = np.random.choice(valid_users, 5, replace=False)
# Kullanıcıları iki gruba ayır
short_duration_users = selected_users[:2]  # İlk 2 kullanıcı kısa süre
long_duration_users = selected_users[2:]   # Son 2 kullanıcı uzun süre

# Anomalileri kaydetmek için index listesi
anomalous_indices = []

# Kısa süreli anomaliler için kullanıcıları işle
for user in short_duration_users:
    user_logs = df[df["UserID"] == user]
    n_anomalies = max(1, int(len(user_logs) * 0.06))
    anomaly_rows = user_logs.sample(n=n_anomalies, random_state=42)
    
    indices = anomaly_rows.index
    df.loc[indices, "AccessDuration"] = np.random.randint(10, 61, size=n_anomalies)  # 10-60 saniye
    anomalous_indices.extend(indices)

# Uzun süreli anomaliler için kullanıcıları işle
for user in long_duration_users:
    user_logs = df[df["UserID"] == user]
    n_anomalies = max(1, int(len(user_logs) * 0.06))
    anomaly_rows = user_logs.sample(n=n_anomalies, random_state=42)
    
    indices = anomaly_rows.index
    df.loc[indices, "AccessDuration"] = np.random.randint(4000, 6001, size=n_anomalies)  # 4000-6000 saniye
    anomalous_indices.extend(indices)

# Anormal satırları ayrı kaydet
anomalies_df = df.loc[anomalous_indices]
anomalies_df.to_csv(anomalies_path, mode='a', index=False, header=not os.path.exists(anomalies_path))

# Güncellenmiş tüm veriyi kaydet
df.to_csv(output_path, index=False)

# Değişen satırların ID'lerini yazdır
print("Anomali uygulanan satırların ID'leri:")
print(df.loc[anomalous_indices, "ID"].tolist())

# --- Anomalous UserID'leri kayıt altına al ---
anomalous_user_ids = df.loc[anomalous_indices, "UserID"].unique()
user_tracking_path = r"DatasetGenerator\GeneratingSyntheticLogDatas\Fourth\test\Test_abnormal_users.txt"

# Varsa mevcut kullanıcıları oku
if os.path.exists(user_tracking_path):
    with open(user_tracking_path, "r") as f:
        existing_users = set(f.read().splitlines())
else:
    existing_users = set()

# Yeni anomalous kullanıcıları ekle
updated_users = existing_users.union(set(anomalous_user_ids.astype(str)))

# Dosyaya tekrar yaz
with open(user_tracking_path, "w") as f:
    for uid in sorted(updated_users):
        f.write(uid + "\n")
