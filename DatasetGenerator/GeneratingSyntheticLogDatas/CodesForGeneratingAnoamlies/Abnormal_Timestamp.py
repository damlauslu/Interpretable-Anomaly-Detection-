import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random

# Dosya yolları
input_path = r"DatasetGenerator\GeneratingSyntheticLogDatas\Fourth\test\Test.csv"
output_path = r"DatasetGenerator\GeneratingSyntheticLogDatas\Fourth\test\Test.csv"
anomaly_path = r"DatasetGenerator\GeneratingSyntheticLogDatas\Fourth\test\Test_anomalous_timestamp.csv"
anomalous_users_path = r"DatasetGenerator\GeneratingSyntheticLogDatas\Fourth\test\Test_abnormal_users.txt"

# Veriyi oku
df = pd.read_csv(input_path)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Yeni liste oluştur: gece ve gündüz kullanıcıları (baskın saat aralığına göre)
night_users = []
day_users = []

for user_id, group in df.groupby("UserID"):
    hours = group["Timestamp"].dt.hour
    total = len(hours)
    night_count = sum((hours >= 20) | (hours <= 6))
    day_count = sum((hours >= 8) & (hours <= 18))

    if total == 0:
        continue
    if night_count / total >= 0.6:
        night_users.append(user_id)
    elif day_count / total >= 0.6:
        day_users.append(user_id)

# Önceden anomali uygulanmış kullanıcıları yükle
if os.path.exists(anomalous_users_path):
    with open(anomalous_users_path, "r") as f:
        existing_anomalous_users = set(line.strip() for line in f.readlines())
else:
    existing_anomalous_users = set()

# Uygun kullanıcıları seç
valid_night_users = list(set(night_users) - existing_anomalous_users)
valid_day_users = list(set(day_users) - existing_anomalous_users)

# Seçim
if len(valid_night_users) < 1 or len(valid_day_users) < 2:
    raise ValueError("Yeterli sayıda uygun gece/gündüz kullanıcısı yok.")

selected_night = np.random.choice(valid_night_users, 2, replace=False)[0]
selected_days = np.random.choice(valid_day_users, 2, replace=False)

anomalous_indices = []

# Gece kullanıcısının 3 logunu gündüz saatine taşı (09:00–17:00)
night_logs = df[df["UserID"] == selected_night]
if len(night_logs) < 3:
    raise ValueError(f"{selected_night} kullanıcısında yeterli log yok.")
night_logs = night_logs.sample(n=5, random_state=42)

for idx in night_logs.index:
    new_hour = random.randint(9, 17)
    new_minute = random.randint(0, 59)
    new_second = random.randint(0, 59)
    ts = df.at[idx, "Timestamp"]
    df.at[idx, "Timestamp"] = ts.replace(hour=new_hour, minute=new_minute, second=new_second)
    anomalous_indices.append(idx)

# Gündüz kullanıcılarının her birinden 3 logu gece saatine taşı (00:00–05:00)
for user in selected_days:
    user_logs = df[df["UserID"] == user]
    if len(user_logs) < 3:
        raise ValueError(f"{user} kullanıcısında yeterli log yok.")
    user_logs = user_logs.sample(n=5, random_state=42)
    for idx in user_logs.index:
        new_hour = random.randint(0, 6)
        new_minute = random.randint(0, 59)
        new_second = random.randint(0, 59)
        ts = df.at[idx, "Timestamp"]
        df.at[idx, "Timestamp"] = ts.replace(hour=new_hour, minute=new_minute, second=new_second)
        anomalous_indices.append(idx)

# Anomaliyi ayrı dosyaya yaz
df.loc[anomalous_indices].to_csv(anomaly_path, mode='a', index=False, header=not os.path.exists(anomaly_path))

# Güncellenmiş veri setini kaydet
df.to_csv(output_path, index=False)

# Anomali uygulanan kullanıcıları kaydet
with open(anomalous_users_path, "a") as f:
    f.write(f"{selected_night}\n")
    for user in selected_days:
        f.write(f"{user}\n")

# Sonuç
print("Anomali uygulanan satırların ID'leri:")
print(df.loc[anomalous_indices, "ID"].tolist())
