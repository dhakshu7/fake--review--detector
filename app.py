import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Title
st.title("🕵️ Fake Review Detector")

# ⭐ Rating Slider
rating = st.slider("Select Rating", 1, 5, 3)

# 📝 Review Input
review = st.text_area("Enter Review")

# 🔘 Button
if st.button("Check"):

    if review.strip() == "":
        st.warning("Please enter a review")

    else:
        # Combine rating + review
        input_text = str(rating) + " " + review

        # Transform input
        vec = vectorizer.transform([input_text])

        # Predict
        prediction = model.predict(vec)[0]

        # Probability
        prob = model.predict_proba(vec)[0]

        # Output
        if prediction == 1:
            st.error(f"⚠️ Fake Review Detected ({prob[1]*100:.2f}% confidence)")
        else:
            st.success(f"✅ Genuine Review ({prob[0]*100:.2f}% confidence)")