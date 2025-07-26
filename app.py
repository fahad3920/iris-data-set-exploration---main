import streamlit as st
import pickle
import numpy as np

# Load trained model and scaler
with open("iris_voting_clf.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("iris_scaler_voting_clf.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Class labels
label_map = {
    0: 'Iris-setosa',
    1: 'Iris-versicolor',
    2: 'Iris-virginica'
}

# Page setup
st.set_page_config(page_title="🌸 Iris Flower Classifier", page_icon="🌼", layout="centered")
st.title("🌸 Iris Flower Classifier")
st.subheader("Using Voting Classifier with Scaled Inputs")

# Input fields
st.markdown("### 🌿 Enter the flower's measurements below:")

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# Calculate areas
sepal_area = sepal_length * sepal_width
petal_area = petal_length * petal_width

# Prepare input for model
features = np.array([[sepal_length, sepal_width, petal_length, petal_width, petal_area, sepal_area]])
features_scaled = scaler.transform(features)

# Predict
if st.button("🔍 Predict Species"):
    prediction = model.predict(features_scaled)[0]
    probs = model.predict_proba(features_scaled)[0]

    st.success(f"🌼 Predicted Species: **{label_map[prediction]}**")

    st.markdown("### 🔢 Prediction Probabilities:")
    for i in range(3):
        st.write(f"- {label_map[i]}: **{probs[i]:.2%}**")

# Footer
st.markdown("---")
st.caption("Built with ❤️ by Munib | Powered by VotingClassifier & Streamlit")
