import pandas as pd
import random

# CSV dosyasını yükleme
df = pd.read_csv(r"./DatasetGenerator/GeneratingSyntheticLogDatas/Fourth/hospital_access_logs.csv")

# Kaç satırın değişeceğini hesapla (%20)
num_rows = len(df)
num_to_modify = int(num_rows * 0.20)

# Rastgele hangi satırların değişeceğini seç
rows_to_modify = random.sample(range(num_rows), num_to_modify)

# Seçilen satırların Department kolonunu değiştir
for idx in rows_to_modify:
    df.at[idx, 'Department'] = random.choice(["DPT_5", "DPT_8"])

df.to_csv(r"./DatasetGenerator/GeneratingSyntheticLogDatas/Fourth/hospital_access_logs.csv", index=False)
print("Updated!")
