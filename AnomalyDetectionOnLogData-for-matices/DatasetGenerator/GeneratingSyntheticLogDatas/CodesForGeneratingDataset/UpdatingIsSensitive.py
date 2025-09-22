import pandas as pd

# CSV dosyasını yükleme
df = pd.read_csv(r"./DatasetGenerator/GeneratingSyntheticLogDatas/Fourth/hospital_access_logs.csv")

# Nurse ve Secretary rollerine sahip kullanıcıların IsSensitive değerlerini 0 yapma
df.loc[df['UserRole'].isin(['Nurse', 'Secretary']), 'IsSensitive'] = 0

# Güncellenmiş CSV'yi kaydetme
df.to_csv(r"./DatasetGenerator/GeneratingSyntheticLogDatas/Fourth/hospital_access_logs.csv", index=False)

print("Nurse ve Secretary için IsSensitive sütunu sıfırlandı ve dosya güncellendi!")