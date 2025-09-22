# frontend/api_utils.py

import requests
import pandas as pd

def post_preprocess_api(df):
    url = "http://localhost/preprocess"   # NGINX üzerinden
    data = {"data": df.to_dict(orient="records")}
    try:
        resp = requests.post(url, json=data, timeout=20)
        if resp.status_code == 200:
            result = resp.json()
            clean_df = pd.DataFrame(result["data"])
            return clean_df, None
        else:
            return None, f"Hata: {resp.text}"
    except Exception as e:
        return None, f"Bağlantı hatası: {e}"
