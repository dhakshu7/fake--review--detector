import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page Configuration
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🕵️",
    layout="centered"
)

# Title
st.title("🕵️ Fake Review Detector")
st.write("Detect whether an online product review is **Genuine** or **Fake** using Machine Learning.")

# Review Input
review = st.text_area(
    "📝 Enter Review",
    placeholder="Example: This product is amazing and worth every penny!"
)

# Check Button
if st.button("🔍 Check Review", use_container_width=True):

    if review.strip() == "":
        st.warning("⚠️ Please enter a review.")
    else:
        # Convert review into vector
        vec = vectorizer.transform([review])

        # Prediction
        prediction = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        # Display Result
        if prediction == 1:
            st.error(f"🚨 Fake Review Detected\n\nConfidence: {prob[1]*100:.2f}%")
        else:
            st.success(f"✅ Genuine Review\n\nConfidence: {prob[0]*100:.2f}%")