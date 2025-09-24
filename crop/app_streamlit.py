import streamlit as st
import numpy as np
import pickle
import os

# ------------------- Load Trained Model -------------------
MODEL_PATH = os.path.join("artifacts", "model.pkl")

model = None
label_encoder = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)

    # Handle dict or direct object
    if isinstance(model_data, dict):
        model = model_data["model"]
        label_encoder = model_data.get("encoder", None)
    else:
        model = model_data
else:
    st.error("⚠️ Model file not found! Please upload `artifacts/model.pkl`.")

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="🌾 Crop Recommendation System", page_icon="🌱", layout="wide")

st.title("🌾 Smart Crop Recommendation System")
st.markdown("""
This system recommends the **best crop to cultivate** based on soil and environmental conditions.
Fill in the parameters below and get a recommendation instantly. 🚀
""")

# Sidebar for inputs
st.sidebar.header("🧑‍🌾 Input Parameters")
N = st.sidebar.number_input("Nitrogen (N)", min_value=0.0, step=0.1, value=50.0)
P = st.sidebar.number_input("Phosphorus (P)", min_value=0.0, step=0.1, value=50.0)
K = st.sidebar.number_input("Potassium (K)", min_value=0.0, step=0.1, value=50.0)
temperature = st.sidebar.number_input("Temperature (°C)", min_value=-10.0, step=0.1, value=25.0)
humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1, value=60.0)
ph = st.sidebar.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.1, value=6.5)
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, step=0.1, value=200.0)

# Predict Button
if st.sidebar.button("🌱 Recommend Crop"):
    if model is None:
        st.error("⚠️ No trained model available. Please upload `artifacts/model.pkl`.")
    else:
        try:
            features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            pred_class = model.predict(features)[0]

            # Handle output type
            if isinstance(pred_class, str):
                prediction = pred_class
            elif label_encoder:
                prediction = label_encoder.inverse_transform([int(pred_class)])[0]
            else:
                prediction = str(pred_class)

            # Display result in a styled card
            st.success(f"✅ Recommended Crop: **{prediction.upper()}** 🌱")

            # Add a little celebratory animation
            st.balloons()

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")

# Footer
st.markdown("----")
st.markdown("🔬 Built with ❤️ using **Streamlit** and **Machine Learning**") 

