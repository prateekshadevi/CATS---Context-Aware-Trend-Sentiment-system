import streamlit as st
import pandas as pd
import os
import getpass

# 1. SETUP & THEME
st.set_page_config(page_title="CATS Intelligence Feed", layout="wide")
st.markdown("""<style>
.main { background-color: #0E1117; color: #FFFFFF; }
[data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
.stExpander { border: 1px solid #30363D !important; background-color: #1E1E1E !important; }
</style>""", unsafe_allow_html=True)

# 2. DATA LOADING
user = getpass.getuser()
SCRATCH_PATH = f'/scratch/{user}/'

@st.cache_data
def load_data():
    master_df = pd.read_csv(os.path.join(SCRATCH_PATH, 'cats_master_dashboard_data.csv'))
    predict_df = pd.read_csv(os.path.join(SCRATCH_PATH, 'cats_predicted_top10.csv'))
    return master_df, predict_df

try:
    df, df_pred = load_data()
except Exception as e:
    st.error(f"Waiting for CSV files in {SCRATCH_PATH}...")
    st.stop()

# 3. HEADER
st.title("🗠 CATS: Context-Aware Trend Sentiment")
st.caption(f"🚀 **System Dashboard**")
st.divider()

# 4. SIDEBAR
st.sidebar.title("📡 System Control")
min_integrity = st.sidebar.slider("Minimum Scientific Integrity", 0.0, 1.0, 0.4, 0.05)

# 5. TABS
tab1, tab2 = st.tabs(["🎯 Today's Intelligence Feed", "🔮 Horizon Forecast"])

with tab1:
    filtered_df = df[df['system_integrity'] >= min_integrity]
    if filtered_df.empty:
        st.warning(f"No trends meet the Integrity threshold of {min_integrity}")
    else:
        for _, row in filtered_df.iterrows():
            # Formatting logic
            disp_label = str(row.get('Display Label', '⚪ Neutral'))
            topic_name = str(row['topic']).upper()
            integrity_val = max(0.0, min(float(row.get('system_integrity', 0)), 1.0))

            with st.expander(f"📌 {topic_name} | {disp_label}"):
                col_text, col_metrics = st.columns([2, 1.2])
                with col_text:
                    st.markdown("### 💡 Intelligence Summary")
                    st.write(row['generated_summary'])
                    st.caption(f"📂 **Sources:** {row.get('sources_used', 'N/A')} | 🧩 **Chunks:** {row.get('chunks_used', 'N/A')}")
                with col_metrics:
                    st.markdown("### 🔬 Scientific Audit")
                    m1, m2 = st.columns(2)
                    m1.metric("SBERT", f"{row.get('sbert_vs_context', 0):.2f}")
                    m2.metric("Faithfulness", f"{row.get('faithfulness', 0):.2f}")
                    st.progress(integrity_val, text=f"System Integrity: {integrity_val:.2f}")
                    st.divider()
                    m3, m4 = st.columns(2)
                    m3.metric("Survival", f"{row.get('survival_chance_pct', 0)}%")
                    m4.write(f"**Velocity:**\n{row.get('velocity', 'Stable')}")
                st.success(f"**Forecast:** {row.get('prediction_label', 'Active Trend')}")

with tab2:
    st.header("🔮 Predicted Top 10 (Next 24h)")
    st.write("Calculated based on social momentum, volume growth, and survival probability.")
    # Add column_config to force the progress bar
    st.dataframe(
        df_pred[['predicted_rank', 'topic', 'survival_chance_pct', 'prediction_label', 'momentum_signal', 'seen_yesterday']], 
        column_config={
            "predicted_rank": "Rank",
            "topic": "Trend Name",
            "survival_chance_pct": st.column_config.ProgressColumn(
                "Survival Chance",
                help="Probability of this trend remaining in the Top 10 tomorrow",
                format="%f%%",
                min_value=0,
                max_value=100,
            ),
            "prediction_label": "Class",
            "momentum_signal": "Pulse",
            "seen_yesterday": "Seen Yesterday"
        },
        hide_index=True, 
        use_container_width=True
    )
