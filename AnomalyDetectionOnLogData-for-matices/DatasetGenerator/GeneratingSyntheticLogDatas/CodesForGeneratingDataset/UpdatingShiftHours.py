import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Mevcut veri setini oku
df = pd.read_csv(r"./DatasetGenerator/GeneratingSyntheticLogDatas/Fourth/hospital_access_logs.csv")

# Rastgele yarısını seç ve gruplara ayır
np.random.seed(42)
random.seed(42)

# Kullanıcı rollerine göre eşit şekilde ayırmak için
roles = ["Doctor", "Nurse", "Secretary", "Admin", "Researcher"]
day_shift_users = []
night_shift_users = []

for role in roles:
    role_users = df[df["UserRole"] == role]["UserID"].unique().tolist()
    random.shuffle(role_users)
    half = len(role_users) // 2
    day_shift_users.extend(role_users[:half])
    night_shift_users.extend(role_users[half:])

# Yeni zamanları belirle
for index, row in df.iterrows():
    if row["UserID"] in day_shift_users:
        new_hour = random.randint(9, 17)  # 09:00 - 18:00
    else:
        new_hour = random.choice(list(range(19, 24)) + list(range(0, 7)))  # 20:00 - 07:00
    
    new_minute = random.randint(0, 59)
    new_second = random.randint(0, 59)
    new_timestamp = datetime.strptime(row["Timestamp"], "%Y-%m-%d %H:%M:%S")
    new_timestamp = new_timestamp.replace(hour=new_hour, minute=new_minute, second=new_second)
    df.at[index, "Timestamp"] = new_timestamp.strftime("%Y-%m-%d %H:%M:%S")

# Güncellenmiş veri setini kaydet
df.to_csv(r"./DatasetGenerator/GeneratingSyntheticLogDatas/Fourth/hospital_access_logs.csv", index=False)
print("Log verileri vardiya sistemine göre güncellendi ve kaydedildi!")