import pandas as pd
import numpy as np
import os

# Dosya yolları
input_path = r"DatasetGenerator\GeneratingSyntheticLogDatas\Fourth\test\Test.csv"
output_path = r"DatasetGenerator\GeneratingSyntheticLogDatas\Fourth\test\Test.csv"
anomalies_path = r"DatasetGenerator\GeneratingSyntheticLogDatas\Fourth\test\Test_anomalous_connection.csv"
user_tracking_path = r"DatasetGenerator\GeneratingSyntheticLogDatas\Fourth\test\Test_abnormal_users.txt"

# Veriyi yükle
df = pd.read_csv(input_path)

# Daha önce anomalilere dahil edilmiş kullanıcıları oku
if os.path.exists(user_tracking_path):
    with open(user_tracking_path, "r") as f:
        used_user_ids = set(f.read().splitlines())
else:
    used_user_ids = set()

# Sadece OnSite erişimi olan ve daha önce anomaly yapılmamış kullanıcıları seç
onsite_users = (
    df[df["Connection"] == "OnSite"]
    .groupby("UserID")
    .filter(lambda x: len(x) > 5)["UserID"]
    .unique()
)

eligible_users = [uid for uid in onsite_users if uid not in used_user_ids]

# Rastgele 3 kullanıcı seç
if len(eligible_users) < 3:
    selected_users = eligible_users  # yetersiz kullanıcı varsa hepsini al
else:
    selected_users = np.random.choice(eligible_users, 3, replace=False)

anomalous_indices = []

# Seçilen kullanıcıların bazı loglarının Connection'ını VPN yap
for user in selected_users:
    user_logs = df[(df["UserID"] == user) & (df["Connection"] == "OnSite")]
    n_anomalies = max(1, int(len(user_logs) * 0.03))
    anomaly_rows = user_logs.sample(n=n_anomalies, random_state=42)
    
    indices = anomaly_rows.index
    df.loc[indices, "Connection"] = "VPN"
    anomalous_indices.extend(indices)

# Anormal satırları ayrı kaydet
anomalies_df = df.loc[anomalous_indices]
anomalies_df.to_csv(anomalies_path, mode='a', index=False, header=not os.path.exists(anomalies_path))

# Güncellenmiş tüm veriyi kaydet
df.to_csv(output_path, index=False)

# Değişen satırların ID'lerini yazdır
print("VPN Anomali uygulanan satırların ID'leri:")
print(df.loc[anomalous_indices, "ID"].tolist())

# --- Anomalous UserID'leri kayıt altına al ---
anomalous_user_ids = df.loc[anomalous_indices, "UserID"].unique()

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