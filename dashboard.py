import streamlit as st
import pandas as pd
import os

import streamlit as st

st.set_page_config(page_title="CATS Analytics", layout="wide")

# This CSS forces the metric labels and values to be readable
st.markdown("""
    <style>
    /* Main Metric Box Styling */
    [data-testid="stMetric"] {
        background-color: #1E1E1E !important; /* Deep charcoal background */
        border: 1px solid #3E3E3E !important; /* Subtle border */
        border-left: 5px solid #00FFAA !important; /* Cool Neon Green accent border */
        padding: 15px !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
    }

    /* Metric Value (The Big Numbers) */
    [data-testid="stMetricValue"] {
        color: #00FFAA !important; /* Neon Green for numbers */
        font-family: 'Courier New', monospace !important;
    }

    /* Metric Label (The Text Above) */
    [data-testid="stMetricLabel"] {
        color: #AAAAAA !important; /* Soft grey for labels */
        font-weight: bold !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    /* Make sure the main app background stays dark */
    .main {
        background-color: #0E1117;
    }
    </style>
    """, unsafe_allow_html=True)

# dynamic user path
user = os.environ.get("USER")
SCRATCH_CSV = f"/scratch/{user}/cats_master_dashboard_data.csv"
st.set_page_config(page_title="CATS Intelligence Command", layout="wide", page_icon="🐱")



# Professional Styling

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; border-radius: 10px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🐱 CATS: Context-Aware Trend Sentiment")
st.caption("2026 Social Media Intelligence & Scientific Audit Dashboard")

if not os.path.exists(SCRATCH_CSV):
    st.error("⚠️ Dashboard data not found. Run the integration step in CATS.ipynb.")
else:
    df = pd.read_csv(SCRATCH_CSV)

    # --- SIDEBAR: CONTROLS ---
    st.sidebar.header("🛡️ Audit Filters")
    search_query = st.sidebar.text_input("🔍 Search Topic", "")
    min_integrity = st.sidebar.slider("Min System Integrity", 0.0, 1.0, 0.40)
    selected_sent = st.sidebar.multiselect("Sentiment Filter", 
                                           df['Display Label'].unique(), 
                                           default=df['Display Label'].unique())

    # Apply Filtering
    mask = (df['system_integrity'] >= min_integrity) & \
           (df['Display Label'].isin(selected_sent))

    if search_query:
        mask = mask & (df['topic'].str.contains(search_query, case=False))
    f_df = df[mask]



    # --- TOP LEVEL METRICS (THE TRAFFIC LIGHTS) ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Trends", len(df))
    grounded = len(df[df['audit_flag'].str.contains('Grounded', na=False)])
    m2.metric("Grounded ✅", f"{grounded}", f"{grounded/len(df):.0%} reliable")
    m3.metric("Avg Integrity ⚖️", f"{df['system_integrity'].mean():.2f}")
    m4.metric("Avg Conf 🎯", f"{df['confidence'].mean():.2%}")
    st.divider()

    # --- THE TREND FEED ---
    if f_df.empty:
        st.warning("No trends match your current Audit filters.")

    else:
        for _, row in f_df.iterrows():
            with st.expander(f"📌 {row['topic'].upper()} | {row['Display Label']}", expanded=True):
                # Create Tabs for a cleaner look
                tab_summary, tab_audit = st.tabs(["💡 Intelligence Summary", "🔬 Scientific Audit"])

                with tab_summary:
                    col_text, col_stats = st.columns([2, 1])

                    with col_text:
                        st.markdown("**AI Report:**")
                        st.info(row['generated_summary'])

                    with col_stats:
                        st.markdown("**Social Pulse:**")
                        st.write(f"📈 Velocity: `{row['velocity']}`")
                        st.write(f"📡 Sources: `{row['sources_used']}`")
                        st.write(f"🔍 Chunks: `{row['chunks_used']}`")

                with tab_audit:
                    a1, a2 = st.columns([1, 1])
                    with a1:
                        st.markdown("**Hallucination Check**")
                        flag = str(row['audit_flag'])
                        if "Grounded" in flag: st.success(f"STATUS: {flag}")
                        elif "Borderline" in flag: st.warning(f"STATUS: {flag}")
                        else: st.error(f"STATUS: {flag}")
                        st.caption(f"System Integrity: {row['system_integrity']:.2f}")
                        st.progress(float(row['system_integrity']))

                    with a2:
                        st.markdown("**Core Metrics**")

                        # Show the specific metrics we added
                        st.caption(f"SBERT Similarity: {row['sbert_vs_context']:.4f}")
                        st.caption(f"RAG Faithfulness: {row['faithfulness']:.2f}")
                        st.caption(f"Sentiment Conf: {row['confidence']:.2%}")
