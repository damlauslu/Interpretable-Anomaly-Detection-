import pandas as pd

def load_data(uploaded_file):
    """
    Yüklenen dosyayı uygun formata göre pandas DataFrame olarak okur.
    Noktalı virgül ayracı olan dosyaları da destekler.
    """
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, sep=None, engine='python')  # auto-detect delimiter
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Desteklenmeyen dosya formatı."
        return df, None
    except Exception as e:
        return None, f"Dosya okunurken hata oluştu: {str(e)}"