import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Prediksi Diabetes", layout="centered")

st.title("🩺 Aplikasi Prediksi Diabetes")
st.write("Masukkan data pasien untuk memprediksi diabetes")

# =========================
# INPUT USER
# =========================
glucose = st.number_input("Glucose", 0, 300, 120)
blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
bmi = st.number_input("BMI", 0.0, 60.0, 25.0)
age = st.number_input("Age", 1, 120, 30)

# =========================
# MODEL SEDERHANA
# =========================
X = np.array([
    [120, 70, 25, 30],
    [150, 80, 30, 45],
    [100, 65, 22, 25],
    [180, 90, 35, 55]
])

y = np.array([0, 1, 0, 1])

model = LogisticRegression()
model.fit(X, y)

if st.button("Prediksi"):
    data = np.array([[glucose, blood_pressure, bmi, age]])
    result = model.predict(data)

    if result[0] == 1:
        st.error("⚠️ Berpotensi Diabetes")
    else:
        st.success("✅ Tidak Diabetes")

# =========================
# VISUALISASI
# =========================
st.subheader("📊 Contoh Visualisasi BMI")

bmi_data = [22, 25, 30, 35, 28]
plt.figure()
plt.hist(bmi_data)
st.pyplot(plt)
