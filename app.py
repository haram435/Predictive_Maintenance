import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
from supabase import create_client

# connect to the bridge
url = "https://qxlfygymdhxkhcgwbxhn.supabase.co"
key = "sb_publishable_zqgKoh-Z-oKoaZLxgikFjA_BW3zuaxu"
supabase = create_client(url, key)

# Robust model loading for deployment
@st.cache_resource
def load_model():
    model_path = 'pipeline_L.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

pipeline = load_model()

st.set_page_config(page_title="Predictive Maintenance AI", layout="wide")

st.title("🛠️ Industrial Machine Health Monitor")
st.markdown("---")

if pipeline is None:
    st.error("Model file 'pipeline_L.pkl' not found. Please ensure it is uploaded to your GitHub repository.")
    st.stop()

# Sidebar for User Input
st.sidebar.header("Manual Sensor Inputs")
m_type = st.sidebar.selectbox("Machine Type", ['L', 'M', 'H'], help="L=Low, M=Medium, H=High Quality")
air_temp = st.sidebar.number_input("Air temperature [K]", value=300.0, step=0.1)
proc_temp = st.sidebar.number_input("Process temperature [K]", value=310.0, step=0.1)
rpm = st.sidebar.number_input("Rotational speed [rpm]", value=1500, step=10)
torque = st.sidebar.number_input("Torque [Nm]", value=40.0, step=0.5)
tool_wear = st.sidebar.number_input("Tool wear [min]", value=0, step=1)

if st.sidebar.button("Run Diagnostics"):
    # Create initial DataFrame
    raw_input = pd.DataFrame([{
        'Type': m_type,
        'Air temperature [K]': air_temp,
        'Process temperature [K]': proc_temp,
        'Rotational speed [rpm]': rpm,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear
    }])

    # 1. Feature Engineering
    raw_input['Power_Index'] = raw_input['Torque [Nm]'] * raw_input['Rotational speed [rpm]']
    raw_input['Temp_difference'] = raw_input['Process temperature [K]'] - raw_input['Air temperature [K]']

    
    expected_columns = [
        'Type', 'Air temperature [K]', 'Process temperature [K]', 
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 
        'Temp_difference','Power_Index'
    ]
    
    raw_input = raw_input[expected_columns]

    # 3. Prediction
    try:
        prediction = pipeline.predict(raw_input)[0]
        probabilities = pipeline.predict_proba(raw_input)[0]
        confidence = np.max(probabilities) * 100
        
        # Save to database
        data_to_save = {
            "air_temp": float(air_temp),
            "process_temp": float(proc_temp),
            "rpm": int(rpm),
            "torque": float(torque),
            "prediction": str(prediction),
            "confidence": float(confidence)
        }
        
        try:
            supabase.table("Maintenence_Logs").insert(data_to_save).execute()
            st.toast("✅ Saved to Database!")
        except Exception as e:
            st.sidebar.warning(f"DB Sync Failed: {e}")

        # Display results
        st.subheader("Machine Status Analysis")
        if prediction in ["No Failure", 0, "0"]:
            st.success(f"✅ STATUS: NORMAL (Confidence: {confidence:.2f}%)")
        else:
            st.error(f"🚨 STATUS: {prediction} DETECTED (Confidence: {confidence:.2f}%)")

        st.info(f"Calculated PowerIndex: {raw_input['Power_Index'].values[0]:.2f} | TempDiff: {raw_input['Temp_difference'].values[0]:.2f}K")

        st.write("### Failure Type Probability")
        prob_df = pd.DataFrame({
            'Failure Type': pipeline.classes_,
            'Probability (%)': probabilities * 100
        }).sort_values(by='Probability (%)', ascending=False)
        st.bar_chart(prob_df.set_index('Failure Type'))

    except ValueError as ve:
        st.error(f"Model Error: The features provided don't match what the model expects. Detail: {ve}")

else:
    st.write("👈 Adjust sensor values in the sidebar and click **Run Diagnostics** to begin.")
