import streamlit as st
import pandas as pd
import plotly.express as px

def show_general_dashboard(df):
    st.subheader("📊 General Access Dashboard")

    # --- Top metrics ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Access Count", len(df))
    with col2:
        st.metric("Unique Users", df["User_ID"].nunique())
    with col3:
        if "Access_Timestamp" in df.columns:
            # HERE: dayfirst=True for Turkish style date parsing
            min_date = pd.to_datetime(df['Access_Timestamp'], dayfirst=True).min().strftime("%d.%m.%Y")
            max_date = pd.to_datetime(df['Access_Timestamp'], dayfirst=True).max().strftime("%d.%m.%Y")
            st.metric("Date Range", f"{min_date} - {max_date}")
        else:
            st.info("No Access_Timestamp column")

    # --- Top 5 Users & 3-hour Access Count ---
    st.markdown(" ")
    col4, col5 = st.columns(2)
    with col4:
        if "User_ID" in df.columns and "User_Role" in df.columns:
            top_users = df.groupby(['User_ID', 'User_Role']).size().reset_index(name='Count')
            top_users = top_users.sort_values(by='Count', ascending=False).head(5)
            fig_top_users = px.bar(
                top_users, x="User_ID", y="Count", color="User_Role",
                title="Top 5 Users",
                height=300,
                color_discrete_map={
                    "Hemşire": "#4D96FF", 
                    "Yonetici": "#FFB72B",
                    "Sekreter": "#EA5C5A"
                }
            )
            fig_top_users.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_top_users, use_container_width=True)
        else:
            st.info("Missing User_ID or User_Role column")

    with col5:
        if "Access_Timestamp" in df.columns:
            # *** USE dayfirst=True HERE AS WELL ***
            df['Access_Timestamp'] = pd.to_datetime(df['Access_Timestamp'], dayfirst=True)
            df['3h_slot'] = df['Access_Timestamp'].dt.floor('3H')
            slot_range = pd.date_range(
                start=df['3h_slot'].min().floor('D'),
                end=df['3h_slot'].max().ceil('D') - pd.Timedelta(hours=0),
                freq='3H'
            )
            count_3h = df.groupby('3h_slot').size().reindex(slot_range, fill_value=0).reset_index()
            count_3h.columns = ['3h_slot', 'Access_Count']
            fig_3h = px.line(
                count_3h, x='3h_slot', y='Access_Count',
                title="Access Count (Every 3 Hours, Complete Timeline)",
                height=300
            )
            fig_3h.update_layout(
                xaxis_title="Datetime (every 3 hours)",
                yaxis_title="Access Count",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_3h, use_container_width=True)
        else:
            st.info("No Access_Timestamp column")

    # --- Access Level Bar + Avg Duration metric next to it + Sensitive Pie chart ---
    st.markdown(" ")
    col6, col7, col8 = st.columns([3, 1, 2])
    with col6:
        if "Access_Duration" in df.columns and "Access_Level" in df.columns:
            duration_by_level = df.groupby("Access_Level")["Access_Duration"].agg(['count', 'mean']).reset_index()
            fig_duration = px.bar(
                duration_by_level, x="Access_Level", y="mean",
                title="Average Access Duration per Access Level",
                labels={'mean': 'Average Duration (s)'},
                height=300
            )
            fig_duration.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_duration, use_container_width=True)
        else:
            st.info("Missing Access_Duration or Access_Level column")

    with col7:
        if "Access_Duration" in df.columns:
            st.metric("Avg. Access Duration (ms)", f"{df['Access_Duration'].mean():.1f}")

    with col8:
        if "Is_Sensitive" in df.columns:
            sensitive_counts = df["Is_Sensitive"].value_counts().reset_index()
            sensitive_counts.columns = ["Is_Sensitive", "Count"]
            fig_sensitive = px.pie(
                sensitive_counts, values="Count", names="Is_Sensitive",
                title="Sensitive Access Analysis",
                height=300
            )
            fig_sensitive.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_sensitive, use_container_width=True)
        else:
            st.info("No Is_Sensitive column")
