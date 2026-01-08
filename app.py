import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Diabetes Analytics Dashboard",
    page_icon="🩺",
    layout="wide"
)

# ===============================
# CSS STYLE
# ===============================
st.markdown("""
<style>
body { background-color: #f5f7fb; }
h1, h2, h3 { color: #0d6efd; }
.stButton>button {
    background-color: #0d6efd;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("Diabetes_Final_Data_V2.csv")
    np.random.seed(42)
    df["year"] = np.random.randint(2020, 2027, len(df))
    return df

df = load_data()

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("🩺 Dashboard Menu")

menu = st.sidebar.radio(
    "Navigasi",
    ["📊 Dataset", "📈 Visualisasi", "🤖 Machine Learning", "🧪 Prediksi"]
)

tahun_awal, tahun_akhir = st.sidebar.slider(
    "📅 Rentang Tahun", 2020, 2026, (2020, 2026)
)

gender_filter = st.sidebar.multiselect(
    "👤 Gender",
    df["gender"].unique(),
    default=df["gender"].unique()
)

df_filtered = df[
    (df["year"] >= tahun_awal) &
    (df["year"] <= tahun_akhir) &
    (df["gender"].isin(gender_filter))
]

# ===============================
# HEADER
# ===============================
st.title("📊 Diabetes Analytics Dashboard")
st.caption("Analisis, Visualisasi & Machine Learning (2020–2026)")

# ===============================
# DATASET
# ===============================
if menu == "📊 Dataset":
    st.subheader("📄 Dataset Diabetes")
    st.dataframe(df_filtered, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Data", len(df_filtered))
    col2.metric("Diabetes", (df_filtered["diabetic"] == "Yes").sum())
    col3.metric("Non Diabetes", (df_filtered["diabetic"] == "No").sum())

    st.subheader("📌 Statistik Ringkas")
    st.dataframe(df_filtered.describe())

# ===============================
# VISUALISASI
# ===============================
elif menu == "📈 Visualisasi":
    st.subheader("📈 Visualisasi Data")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.countplot(data=df_filtered, x="diabetic", ax=ax)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.countplot(data=df_filtered, x="year", hue="diabetic", ax=ax)
        st.pyplot(fig)

    st.subheader("🔗 Korelasi")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(
        df_filtered.select_dtypes(include=np.number).corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )
    st.pyplot(fig)

# ===============================
# MACHINE LEARNING
# ===============================
elif menu == "🤖 Machine Learning":
    st.subheader("🤖 Perbandingan Algoritma")

    df_ml = df_filtered.copy()
    le = LabelEncoder()

    df_ml["gender"] = le.fit_transform(df_ml["gender"])
    df_ml["diabetic"] = le.fit_transform(df_ml["diabetic"])

    X = df_ml.drop("diabetic", axis=1)
    y = df_ml["diabetic"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = accuracy_score(y_test, pred)

    result_df = pd.DataFrame.from_dict(
        results, orient="index", columns=["Akurasi"]
    )

    st.dataframe(result_df)

    fig, ax = plt.subplots()
    result_df.plot(kind="bar", legend=False, ax=ax)
    ax.set_ylim(0, 1)
    st.pyplot(fig)

# ===============================
# PREDIKSI PASIEN (LENGKAP)
# ===============================
elif menu == "🧪 Prediksi":
    st.subheader("🧪 Prediksi Diabetes Pasien")

    df_ml = df.copy()
    le = LabelEncoder()

    df_ml["gender"] = le.fit_transform(df_ml["gender"])
    df_ml["diabetic"] = le.fit_transform(df_ml["diabetic"])

    X = df_ml.drop("diabetic", axis=1)
    y = df_ml["diabetic"]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    age = st.number_input("Umur", 1, 100, 40)
    gender = st.selectbox("Gender", ["Male", "Female"])
    glucose = st.number_input("Glucose", 50.0, 300.0, 120.0)
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)

    threshold = st.slider(
        "🎯 Threshold Keputusan",
        0.1, 0.9, 0.5, 0.05
    )

    if st.button("🔍 Prediksi"):
        gender_val = 1 if gender == "Male" else 0

        input_data = np.zeros((1, X.shape[1]))
        input_data[0][0] = age
        input_data[0][1] = gender_val
        input_data[0][5] = glucose
        input_data[0][8] = bmi

        proba = model.predict_proba(input_data)[0][1]
        confidence = proba * 100
        prediction = 1 if proba >= threshold else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Probabilitas", f"{proba:.2f}")
        col2.metric("Confidence", f"{confidence:.2f}%")
        col3.metric("Threshold", threshold)

        if prediction == 1:
            st.error("🟥 HASIL: DIABETES")
        else:
            st.success("🟩 HASIL: TIDAK DIABETES")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("📌 Dashboard Analisis Diabetes | Streamlit + Machine Learning")
