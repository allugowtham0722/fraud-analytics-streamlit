import pandas as pd
import streamlit as st
import datetime
import pickle
import base64
import plotly.express as px
import numpy as np

# Set page configuration
st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title="UPI Fraud Detector", page_icon="💳")

# --- MOCK MODEL LOADING ---
try:
    class MockModel:
        """Simulates the prediction outcome for demonstration purposes."""
        def predict(self, df):
            if isinstance(df, pd.DataFrame):
                return np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
            return np.random.choice([0, 1], size=1, p=[0.9, 0.1])
    loaded_model = MockModel()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# --- FUNCTION TO LOAD LOCAL IMAGE AND ENCODE ---
def get_base64_of_bin_file(bin_file):
    """ Reads a binary file and returns its base64 encoded string. """
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg_from_local_file(file_path):
    """
    Sets the background of the Streamlit app from a local file.
    Includes a dark overlay for text readability.
    """
    try:
        base64_string = get_base64_of_bin_file(file_path)
        st.markdown(
             f"""
             <style>
             /* Target the main container of the Streamlit app */
             [data-testid="stAppViewContainer"] > .main {{
                 background: linear-gradient(rgba(15, 23, 42, 0.9), rgba(15, 23, 42, 0.95)),
                             url("data:image/jpeg;base64,{base64_string}");
                 background-size: cover;
                 background-position: center center;
                 background-repeat: no-repeat;
                 background-attachment: fixed;
             }}

             /* Sidebar Styling */
             [data-testid="stSidebar"] {{
                 background-color: rgba(15, 23, 42, 0.8) !important;
                 backdrop-filter: blur(5px);
             }}
            
             /* General styling for visibility */
             h1, h3, h4, h6, p, label, .st-emotion-cache-16idsys p {{
                 color: #e2e8f0 !important;
             }}
             h1 {{ color: #93c5fd !important; text-align: center; }}
             h3, h4 {{ color: #a5b4fc !important; border-left: 5px solid #6366f1; padding-left: 10px; }}
             .st-expander, .st-container, div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] > div {{
                 background-color: rgba(30, 41, 59, 0.85);
                 padding: 20px !important;
                 border-radius: 15px;
                 border: 1px solid #334155;
                 margin-bottom: 15px;
             }}
             .metric-card {{
                 background-color: #1e293b; padding: 15px; border-radius: 10px;
                 text-align: center; border-left: 5px solid;
             }}
             .metric-card p {{ margin: 0; font-size: 0.8rem; font-weight: bold; }}
             .metric-card h3 {{ border-left: none; margin: 0; padding-left: 0; font-size: 1.8rem; }}
             </style>
             """,
             unsafe_allow_html=True
         )
    except FileNotFoundError:
        st.error("Background image not found. Please make sure assets/background.jpg exists.")


# --- Apply the background at the start of the app ---
# Make sure you have a folder named 'assets' with 'background.jpg' inside it.
set_bg_from_local_file('assets/background.jpg')


# --- CORE FUNCTIONS ---
def preprocess_input(df):
    df_copy = df.copy()
    if 'timestamp' in df_copy.columns:
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], errors='coerce')
        df_copy['Month'] = df_copy['timestamp'].dt.month
        df_copy['Year'] = df_copy['timestamp'].dt.year
    else:
        df_copy['Month'] = datetime.datetime.now().month
        df_copy['Year'] = datetime.datetime.now().year
    if 'is_fraud' in df_copy.columns:
        df_copy = df_copy.drop(columns=['is_fraud'])
    if 'Amount' in df_copy.columns and 'amount' not in df_copy.columns:
        df_copy = df_copy.rename(columns={'Amount': 'amount'})
    if 'amount' not in df_copy.columns:
        df_copy['amount'] = np.random.uniform(50, 5000, len(df_copy))
    if 'vpa_sender' not in df_copy.columns:
        df_copy['vpa_sender'] = [f"user_{np.random.randint(1, 100)}@bank" for _ in range(len(df_copy))]
    if 'vpa_receiver' not in df_copy.columns:
        df_copy['vpa_receiver'] = [f"merchant_{np.random.randint(1, 50)}@pay" for _ in range(len(df_copy))]
    if 'device_id' not in df_copy.columns:
        df_copy['device_id'] = [f"dev_{np.random.randint(1, 20)}" for _ in range(len(df_copy))]
    for col in ['vpa_sender', 'vpa_receiver', 'device_id']:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype('object')
    expected_features = ['amount', 'vpa_sender', 'vpa_receiver', 'device_id', 'Month', 'Year']
    for feature in expected_features:
        if feature not in df_copy.columns:
            df_copy[feature] = 0
    return df_copy[expected_features]

def download_csv_link(df, filename):
    csv = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    href = f"""
    <div class="mt-4" style="text-align: center;">
        <a href="data:file/csv;base64,{b64}" download="{filename}" 
            style="display: inline-block; padding: 10px 20px; border-radius: 8px; 
                   background-color: #10b981; color: white; text-decoration: none; 
                   font-weight: bold; margin-top: 15px; font-size: 1.1rem;">
            📥 Download Results CSV
        </a>
    </div>"""
    st.markdown(href, unsafe_allow_html=True)

def generate_dashboard(df_results):
    st.subheader("📊 Prediction Dashboard")
    total_txns = len(df_results)
    df_results['predicted_fraud'] = df_results['predicted_fraud'].astype(int)
    fraud_count = df_results['predicted_fraud'].sum()
    non_fraud_count = total_txns - fraud_count
    fraud_amount = df_results[df_results['predicted_fraud'] == 1]['amount'].sum()
    st.markdown("<h4 style='color: #a5b4fc; border-bottom: 2px solid #a5b4fc; padding-bottom: 10px; margin-top: 25px;'>Key Fraud & Transaction Indicators</h4>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card" style="border-left-color: #3b82f6;">
            <p style="color: #93c5fd;">TOTAL TRANSACTIONS</p>
            <h3 style="color: #3b82f6;">{total_txns:,.0f}</h3></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card" style="border-left-color: #ef4444;">
            <p style="color: #fca5a5;">FRAUD COUNT</p>
            <h3 style="color: #ef4444;">{fraud_count:,.0f}</h3></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card" style="border-left-color: #f59e0b;">
            <p style="color: #fcd34d;">POTENTIAL LOSS (USD)</p>
            <h3 style="color: #f59e0b;">${fraud_amount:,.2f}</h3></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card" style="border-left-color: #10b981;">
            <p style="color: #a7f3d0;">SAFE TRANSACTIONS</p>
            <h3 style="color: #10b981;">{non_fraud_count:,.0f}</h3></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #a5b4fc; border-bottom: 2px solid #a5b4fc; padding-bottom: 10px; margin-top: 25px;'>Detailed Transaction Distribution</h4>", unsafe_allow_html=True)
    col_vis1, col_vis2 = st.columns(2)
    fraud_labels = {1: '🚨 Fraudulent', 0: '✅ Safe'}
    df_results['Fraud Status'] = df_results['predicted_fraud'].map(fraud_labels)
    if total_txns > 0:
        with col_vis1:
            fig_pie = px.pie(df_results, names='Fraud Status', color='Fraud Status',
                             color_discrete_map={'🚨 Fraudulent': '#ef4444', '✅ Safe': '#10b981'},
                             title='Predicted Transaction Status Breakdown', hole=.4, template="plotly_dark")
            fig_pie.update_layout(legend_title_text='', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_vis2:
            fig_hist = px.histogram(df_results, x='amount', title='Distribution of Transaction Amounts',
                                    color_discrete_sequence=['#3b82f6'], marginal="box", template="plotly_dark")
            fig_hist.update_layout(xaxis_title="Amount ($)", yaxis_title="Count", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_hist, use_container_width=True)
    if fraud_count > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #dc2626; border-bottom: 2px solid #dc2626; padding-bottom: 10px; margin-top: 25px;'>⚠ Top Fraudulent Behavioral Indicators</h4>", unsafe_allow_html=True)
        fraud_df = df_results[df_results['predicted_fraud'] == 1]
        col_top1, col_top2, col_top3 = st.columns(3)
        with col_top1:
            st.markdown("<h6>Frequent Senders</h6>", unsafe_allow_html=True)
            st.dataframe(fraud_df['vpa_sender'].value_counts().nlargest(5).reset_index(), use_container_width=True, hide_index=True)
        with col_top2:
            st.markdown("<h6>Frequent Receivers</h6>", unsafe_allow_html=True)
            st.dataframe(fraud_df['vpa_receiver'].value_counts().nlargest(5).reset_index(), use_container_width=True, hide_index=True)
        with col_top3:
            st.markdown("<h6>Compromised Devices</h6>", unsafe_allow_html=True)
            st.dataframe(fraud_df['device_id'].value_counts().nlargest(5).reset_index(), use_container_width=True, hide_index=True)

# --- UI LAYOUT ---
st.title("💳 Advanced UPI Fraud Detector")
st.markdown("""<div style="text-align: center; margin-bottom: 25px; font-size: 1.2rem; color: #a5b4fc;">
    Analyze single transactions or upload a CSV for a comprehensive fraud report.
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://placehold.co/400x120/6366f1/ffffff?text=FRAUD+ANALYTICS", use_container_width=True)
    st.header("About the Model")
    st.info("This AI-powered tool predicts potential fraud in UPI transactions by analyzing financial and behavioral patterns.")
    st.subheader("Model Status")
    st.markdown("- *Model:* FraudGuard v1.2 (Simulated)\n- *Method:* Classification\n- *Features:* amount, vpa, device_id, etc.")
    st.markdown("---")
    st.caption("© 2025 Fraud Analytics Inc.")

with st.container():
    st.subheader("1. Check a Single Transaction")
    col_tran1, col_tran2 = st.columns(2)
    with col_tran1:
        tran_date = st.date_input("Transaction Date", datetime.date.today(), key='single_date')
        amt = st.number_input("Transaction Amount ($)", min_value=0.01, step=10.0, format="%.2f", key='single_amount')
        device_id = st.text_input("Device ID (e.g., device_12)", key='single_device')
    with col_tran2:
        vpa_sender = st.text_input("Sender VPA (e.g., user1@paytm)", key='single_sender')
        vpa_receiver = st.text_input("Receiver VPA (e.g., user2@ybl)", key='single_receiver')
        selected_date = datetime.datetime.combine(tran_date, datetime.time(12, 0, 0))
    st.markdown("---")
    st.subheader("2. Batch Analysis (Upload CSV)")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Analyze Transaction(s) and Generate Report"):
    if uploaded_file is not None:
        try:
            with st.spinner(f"Analyzing transactions..."):
                df_raw = pd.read_csv(uploaded_file)
                df_for_prediction = preprocess_input(df_raw.copy())
                results = loaded_model.predict(df_for_prediction)
                df_results = df_raw.copy()
                df_results['predicted_fraud'] = results
                if 'Amount' in df_results.columns and 'amount' not in df_results.columns:
                    df_results = df_results.rename(columns={'Amount': 'amount'})
                elif 'amount' not in df_results.columns:
                    df_results['amount'] = df_for_prediction['amount']
                for col in ['vpa_sender', 'vpa_receiver', 'device_id']:
                    if col not in df_results.columns:
                        df_results[col] = df_for_prediction[col]
            st.success(f"✅ Batch analysis complete!")
            generate_dashboard(df_results)
            with st.expander("View Full Analysis Results", expanded=False):
                st.dataframe(df_results, use_container_width=True)
            download_csv_link(df_results, "fraud_detection_results.csv")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    elif amt > 0 and vpa_sender and vpa_receiver and device_id:
        input_dict = {"amount": [amt], "vpa_sender": [vpa_sender], "vpa_receiver": [vpa_receiver], "timestamp": [selected_date], "device_id": [device_id]}
        df_single = pd.DataFrame(input_dict)
        with st.spinner("Analyzing transaction..."):
            df_features = preprocess_input(df_single.copy())
            result = loaded_model.predict(df_features)[0]
        st.success("✅ Analysis complete!")
        st.markdown("---")
        st.subheader("Prediction Result")
        if result == 0:
            st.balloons()
            st.markdown("""<div style="background-color: #2c443b; padding: 25px; border-radius: 15px; border: 3px solid #10b981; text-align: center;">
                <h2 style="color: #a7f3d0; font-size: 2rem;">🎉 TRANSACTION IS SAFE!</h2>
                <p style="color: #a7f3d0; font-size: 1.1rem;">The model predicts this transaction is <b>NOT fraudulent</b>.</p></div>""", unsafe_allow_html=True)
        else:
            st.error("🚨 POTENTIAL FRAUD ALERT!")
            st.markdown("""<div style="background-color: #402c30; padding: 25px; border-radius: 15px; border: 3px solid #ef4444; text-align: center;">
                <h2 style="color: #fca5a5; font-size: 2rem;">⚠ HIGH RISK: FRAUDULENT</h2>
                <p style="color: #fca5a5; font-size: 1.1rem;">Immediate review is highly recommended. The model predicts this transaction is <b>FRAUDULENT</b>.</p></div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Show Input Details"):
            st.dataframe(df_single.drop(columns=['timestamp']), use_container_width=True, hide_index=True)
    else:
        st.warning("⚠ Please fill in all fields for a single check or upload a CSV file.")