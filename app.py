import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page Title
st.title("🕵️ Fake Review Detector")

# ⭐ Rating Dropdown
rating = st.selectbox(
    "Select Rating",
    [1, 2, 3, 4, 5]
)

# 📝 Review Input
review = st.text_area("Enter Review")

# 🔘 Check Button
if st.button("Check"):

    if review.strip() == "":
        st.warning("Please enter a review")

    else:
        # Combine rating and review
        input_text = f"{rating} {review}"

        # Convert to vector
        vec = vectorizer.transform([input_text])

        # Predict
        prediction = model.predict(vec)[0]

        # Confidence
        prob = model.predict_proba(vec)[0]

        # Show result
        if prediction == 1:
            st.error(f"⚠️ Fake Review Detected ({prob[1]*100:.2f}% confidence)")
        else:
            st.success(f"✅ Genuine Review ({prob[0]*100:.2f}% confidence)")