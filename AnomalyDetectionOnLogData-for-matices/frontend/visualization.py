import plotly.express as px

def plot_access_duration_histogram(df):
    if "access_duration" in df.columns:
        return px.histogram(df, x="access_duration", nbins=30, title="Access Duration Histogramı")
    return None

def plot_user_access_bar(df):
    if "user_id" in df.columns:
        user_counts = df["user_id"].value_counts().reset_index()
        user_counts.columns = ["user_id", "access_count"]
        return px.bar(user_counts.head(20), x="user_id", y="access_count", title="Kullanıcıya Göre Erişim Sayısı (İlk 20)")
    return None
def plot_time_series(df):
    # "timestamp" ve "access_duration" olan bir dataframe bekler
    if "timestamp" in df.columns and "access_duration" in df.columns:
        df_sorted = df.sort_values("timestamp")
        fig = px.line(df_sorted, x="timestamp", y="access_duration", title="Zaman Serisi (Access Duration)")
        return fig
    return None