import pandas as pd
import random

# CSV dosyasını yükleme
df = pd.read_csv(r"./DatasetGenerator/GeneratingSyntheticLogDatas/Fourth/hospital_access_logs.csv")

# Tüm farklı USR_ID'leri al
unique_ids = df['UserID'].unique()

# Eğer 20'den az farklı ID varsa, hepsini al, yoksa 20 tanesini rastgele seç
num_to_select = min(10, len(unique_ids))
selected_ids = random.sample(list(unique_ids), num_to_select)

# Bu ID'lere sahip satırların Connection kolonunu 'VPN' yap
df.loc[df['UserID'].isin(selected_ids), 'Connection'] = 'VPN'

# Değişiklikleri kaydet
df.to_csv(r"./DatasetGenerator/GeneratingSyntheticLogDatas/Fourth/hospital_access_logs.csv", index=False)
print("Updated!")

# Seçilen ID'leri yazdır
print(f"VPN yapılan USR_ID'ler: {selected_ids}")