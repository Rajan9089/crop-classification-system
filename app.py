import streamlit as st
import numpy as np
import pickle

# Load your model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Title of the web app
st.title("Crop Recommendation System 🌱")

# Create a sidebar for input fields
with st.sidebar:
    st.header("Input Parameters")
    N = st.number_input("Nitrogen content", min_value=0.0, value=0.0, step=0.1, format="%.1f", help="Enter Nitrogen content")
    P = st.number_input("Phosphorus content", min_value=0.0, value=0.0, step=0.1, format="%.1f", help="Enter Phosphorus content")
    K = st.number_input("Potassium content", min_value=0.0, value=0.0, step=0.1, format="%.1f", help="Enter Potassium content")
    temp = st.number_input("Temperature in °C", min_value=0.0, value=0.0, step=0.1, format="%.1f", help="Enter Temperature in °C")
    humidity = st.number_input("Humidity in %", min_value=0.0, value=0.0, step=0.1, format="%.1f", help="Enter Humidity in %")
    ph = st.number_input("pH value", min_value=0.0, value=0.0, step=0.1, format="%.1f", help="Enter pH value")
    rainfall = st.number_input("Rainfall in mm", min_value=0.0, value=0.0, step=0.1, format="%.1f", help="Enter Rainfall in mm")

if st.button("Get Recommendation"):
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = f"{crop} is the best crop to be cultivated right there."
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    st.success(result)

# Footer
st.markdown("""
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            color: #555;
            text-align: center;
            padding: 10px;
        }
    </style>
    <div class="footer">
        Powered by Streamlit
    </div>
    """, unsafe_allow_html=True)