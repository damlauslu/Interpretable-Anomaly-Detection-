import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Sabitler
NUM_RECORDS = 500000
NUM_USERS = 200  # Personel sayısı
NUM_PATIENTS = 20000  # Hasta sayısı
NUM_DEPARTMENTS = 10  # Departman sayısı
NUM_DEVICES = 200  # Cihaz sayısı
USER_ROLES = ["Doctor", "Nurse", "Secretary", "Admin", "Researcher"]
ACCESS_LEVELS = ["read", "write", "modify", "delete"]

# Rastgele veri üretme
np.random.seed(42)
random.seed(42)

# Kullanıcıların departmanlarını belirleyelim
user_departments = {f"USR_{str(user_id).zfill(3)}": random.randint(1, NUM_DEPARTMENTS) 
                    for user_id in range(1, NUM_USERS + 1)}

# Kullanıcı rollerini özel dağılımla belirleme
user_roles = {}
user_distribution = {
    "Doctor": 50,
    "Nurse": 80,
    "Secretary": 30,
    "Admin": 10,
    "Researcher": 30
}
user_id_counter = 1
for role, count in user_distribution.items():
    for _ in range(count):
        user_roles[f"USR_{str(user_id_counter).zfill(3)}"] = role
        user_id_counter += 1

# Hastaların genellikle ziyaret ettiği departmanları belirleyelim
patient_departments = {f"PTN_{str(patient_id).zfill(5)}": random.randint(1, NUM_DEPARTMENTS) 
                       for patient_id in range(1, NUM_PATIENTS + 1)}

# Veri kümesini oluştur
records = []
start_time = datetime(2024, 1, 1, 9, 0, 0)  # 9 AM başlangıç
end_time = start_time + timedelta(days=3)  # 3 gün

for i in range(NUM_RECORDS):
    log_id = i + 1  # ID sütunu ekleme
    user_id = f"USR_{str(random.randint(1, NUM_USERS)).zfill(3)}"
    department_id = f"DPT_{str(user_departments[user_id]).zfill(3)}"
    role = user_roles[user_id]
    
    # 9 AM - 6 PM aralığında rastgele zaman üretme
    random_day = random.randint(0, 2)  # 3 günlük periyot
    random_hour = random.randint(9, 17)  # 9AM - 6PM
    random_minute = random.randint(0, 59)
    random_second = random.randint(0, 59)
    timestamp = start_time + timedelta(days=random_day, hours=random_hour - 9, minutes=random_minute, seconds=random_second)
    
    access_level = np.random.choice(ACCESS_LEVELS, p=[0.7, 0.15, 0.1, 0.05])
    access_duration = np.random.randint(120, 300)  # 120 saniye - 300 saniye arası
    device_id = f"DVC_{str(random.randint(1, NUM_DEVICES)).zfill(3)}"
    patient_id = f"PTN_{str(random.randint(1, NUM_PATIENTS)).zfill(5)}"
    visit_department = f"DPT_{str(patient_departments[patient_id]).zfill(3)}"
    
    connection = "OnSite"
    is_access_fail = np.random.choice([0, 1], p=[0.98, 0.02])  # Başarısız erişimler %2 ihtimalle
    is_sensitive = np.random.choice([0, 1], p=[0.8, 0.2])  # Hassas veri erişimi %20 ihtimalle
    
    # Her zaman hastanın ziyaret ettiği departmandaki personel erişmeli
    department_id = visit_department
    
    records.append([log_id, user_id, department_id, role, connection, timestamp, access_level, access_duration, device_id, 
                    patient_id, is_access_fail, is_sensitive, visit_department])

# DataFrame oluşturma
df = pd.DataFrame(records, columns=["ID", "UserID", "Department", "UserRole", "Connection", "Timestamp", "AccessLevel", 
                                    "AccessDuration", "DeviceID", "PatientID", "IsAccessFail", "IsSensitive", "VisitDepartment"])

# CSV'ye kaydetme
df.to_csv(r"./DatasetGenerator/GeneratingSyntheticLogDatas/Fourth/hospital_access_logs.csv", index=False)
print("Sentetik hastane erişim verisi oluşturuldu ve kaydedildi!")