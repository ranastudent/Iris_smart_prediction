import streamlit as st
import torch
import joblib
import numpy as np
from model import IrisSmartNN 

# ১. রিসোর্স লোডিং (Cache ব্যবহার করা হয়েছে পারফরম্যান্সের জন্য)
@st.cache_resource
def init_resources():
    
    # মডেল কঙ্কাল তৈরি
    model = IrisSmartNN(input_dim=4, hidden_dim=16, output_dim=3, num_layers=2, dropout_rate=0.05)
    # সেভ করা ওয়েট লোড করা
    model.load_state_dict(torch.load("iris_smart_model.pth", map_location="cpu"))
    model.eval() # ইনফারেন্স মোড
    
    # স্কেলার লোড
    scaler = joblib.load("scaler.joblib")
    return model, scaler

# রিসোর্সগুলো এক্সেস করা
try:
    model, scaler = init_resources()
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")

# ২. ইউজার ইন্টারফেস (UI)
st.title("Iris Species Classifier 🌸")
st.markdown("Enter flower measurements to get the prediction.")

st.sidebar.header("Flower Measurements")
sepal_l = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8) # Default value changed to match your test
sepal_w = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 2.5)
petal_l = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.1)
petal_w = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

# ৩. প্রেডিকশন বাটন
if st.button("Predict"):
    # Raw ডাটা প্রসেস
    raw_data = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
    
    # প্রফেশনাল স্কেলিং (ট্রেনিংয়ের স্কেলার দিয়ে)
    scaled_data = scaler.transform(raw_data)
    
    # টেনসরে রূপান্তর
    input_tensor = torch.tensor(scaled_data, dtype=torch.float32)
    
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        
    # সঠিক ক্যাটাগরি লিস্ট (বর্ণানুক্রমিক)
    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    res = classes[pred.item()]
    
    # রেজাল্ট দেখানো
    st.success(f"**Prediction: {res}**")
    
    # কনফিডেন্স স্কোর দেখানো (Professional Touch)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    confidence = probabilities[0][pred.item()].item() * 100
    st.info(f"Confidence Level: {confidence:.2f}%")
