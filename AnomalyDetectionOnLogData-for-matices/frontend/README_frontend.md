Virtual Environment ve Ortak Geliştirme Yönergesi
Bu yönerge, anomaly detection projesi kapsamında ekibin farklı parçaları (frontend, model service, explanation service) üzerinde paralel ve uyumlu şekilde çalışabilmesi için geliştirme ortamının nasıl kurulacağına ve nasıl kullanılacağına dair adımları açıklar.
1. Ortak Çalışma Kuralları

- Her geliştirici kendi klasöründe (`frontend/`, `model_service/`, `explanation_service/`) çalışmalıdır.
- Tüm bağımlılıklar klasöre özel `requirements.txt` dosyası içinde tanımlanmalıdır.
- Her geliştirici kendi ortamında `virtual environment` (sanallaştırılmış Python ortamı) oluşturmalı ve bu ortamı aktif ederek geliştirme yapmalıdır.
- `venv/` klasörleri GIT'e dahil edilmemeli, `.gitignore` dosyasına mutlaka yazılmalıdır.

2. Virtual Environment Kurulumu
Her klasör için aşağıdaki adımları uygulayın:
1. Terminali açın ve çalışacağınız klasöre girin (örneğin `cd frontend`).
2. Ortamı oluşturun:
   - `python -m venv venv`
3. Ortamı aktif hale getirin:
   - macOS/Linux: `source venv/bin/activate`
   - Windows CMD: `venv\Scripts\activate`
   - Windows PowerShell: `venv\Scripts\Activate.ps1`
4. Gerekli kütüphaneleri yükleyin:
   - `pip install -r requirements.txt`
5. Yüklenen ortamı `requirements.txt` ile güncelleyin (geliştirme sonrası):
   - `pip freeze > requirements.txt`
3. Streamlit Frontend Çalıştırma

Frontend geliştiricisi, arayüzü çalıştırmak için aşağıdaki adımları uygular:
1. Ortam aktif olmalı.
2. Ana dosya çalıştırılır:
   - `streamlit run main.py`

4. Model ve Açıklama Servislerini Çalıştırma

Model geliştiricisi ve açıklama geliştiricisi, kendi servislerini şöyle çalıştırır:
1. Ortamı aktif eder.
2. FastAPI servisini çalıştırır:
   - `uvicorn app:app --reload --port 8001` (model)
   - `uvicorn app:app --reload --port 8002` (açıklama)

5. VS Code Kullanımı (Önerilen)

- VS Code açıldığında sağ alttaki Python versiyonuna tıklayıp `venv` klasörünüzü seçin.
- Böylece tüm kod tamamlama ve çalıştırmalar doğru ortamdan olur.

6. GIT ile Çakışmaları Önlemek

- `venv/` klasörünü `.gitignore` dosyasına mutlaka ekleyin.
- GIT'e sadece `requirements.txt` gibi paylaşılabilir dosyaları gönderin.
- Ortam dosyaları ve yerel klasörler GIT'e gönderilmemeli.

