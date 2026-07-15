import streamlit as st
import joblib
import base64
import re
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🕵️",
    layout="centered"
)

# --------------------------------------------------
# BACKGROUND IMAGE
# --------------------------------------------------
def set_background(image_file):
    image_path = Path(image_file)

    if image_path.exists():
        with open(image_path, "rb") as image:
            encoded_image = base64.b64encode(image.read()).decode()

        st.markdown(
            f"""
            <style>
            [data-testid="stAppViewContainer"] {{
                background-image:
                    linear-gradient(
                        rgba(3, 7, 25, 0.55),
                        rgba(3, 7, 25, 0.55)
                    ),
                    url("data:image/png;base64,{encoded_image}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}

            [data-testid="stHeader"] {{
                background: transparent;
            }}

            [data-testid="stMainBlockContainer"] {{
                background: rgba(5, 10, 35, 0.70);
                padding: 35px;
                border-radius: 20px;
                margin-top: 35px;
                backdrop-filter: blur(8px);
            }}

            .stTextArea textarea {{
                background-color: rgba(20, 24, 40, 0.92) !important;
                color: white !important;
                border-radius: 12px !important;
            }}

            .stButton button {{
                width: 100%;
                border-radius: 10px;
                font-size: 17px;
                font-weight: bold;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )


# Change this to background.jpg if your image is JPG
set_background("background.png")

# --------------------------------------------------
# LOAD MODEL AND VECTORIZER
# --------------------------------------------------
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except FileNotFoundError:
    st.error("Model files are missing. Check model.pkl and vectorizer.pkl.")
    st.stop()

# --------------------------------------------------
# INPUT VALIDATION
# --------------------------------------------------
def validate_review(review_text):
    cleaned_text = review_text.strip()

    if cleaned_text == "":
        return False, "Please enter a review."

    # Extract English alphabetic words
    words = re.findall(r"[A-Za-z]+", cleaned_text)

    if len(words) == 0:
        return False, "Please enter a meaningful review using words."

    letters = "".join(words).lower()

    # Too little information
    if len(letters) < 3:
        return False, "Review is too short. Please enter a meaningful review."

    # Reject long single random strings
    if len(words) == 1 and len(words[0]) > 12:
        return False, "Invalid review. Please enter meaningful words."

    # Reject text without vowels
    vowels = sum(letter in "aeiou" for letter in letters)

    if vowels == 0:
        return False, "Invalid review. The entered text does not look meaningful."

    # Reject very low vowel ratio, common in random keyboard text
    vowel_ratio = vowels / len(letters)

    if len(letters) >= 8 and vowel_ratio < 0.15:
        return False, "Invalid review. Please enter a proper product review."

    return True, ""


# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("🕵️ Fake Review Detector")

st.write(
    "Detect whether an online product review is "
    "**Genuine** or **Fake** using Machine Learning."
)

# --------------------------------------------------
# REVIEW INPUT
# --------------------------------------------------
review = st.text_area(
    "📝 Enter Review",
    placeholder="Example: The product quality is good and worth the money."
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if st.button("🔍 Check Review"):

    is_valid, message = validate_review(review)

    if not is_valid:
        st.warning(f"⚠️ {message}")

    else:
        cleaned_review = review.strip()

        # Convert review into TF-IDF features
        review_vector = vectorizer.transform([cleaned_review])

        # If none of the entered words exist in the trained vocabulary
        if review_vector.nnz == 0:
            st.warning(
                "⚠️ Invalid or unknown review text. "
                "Please enter a meaningful product review."
            )

        else:
            prediction = model.predict(review_vector)[0]
            probabilities = model.predict_proba(review_vector)[0]

            genuine_probability = probabilities[0] * 100
            fake_probability = probabilities[1] * 100

            if prediction == 1:
                st.error(
                    f"🚨 Fake Review Detected\n\n"
                    f"Confidence: {fake_probability:.2f}%"
                )
                confidence = fake_probability

            else:
                st.success(
                    f"✅ Genuine Review\n\n"
                    f"Confidence: {genuine_probability:.2f}%"
                )
                confidence = genuine_probability

            st.progress(int(confidence))

            with st.expander("📊 Probability Details"):
                st.write(
                    f"**Genuine Probability:** "
                    f"{genuine_probability:.2f}%"
                )
                st.write(
                    f"**Fake Probability:** "
                    f"{fake_probability:.2f}%"
                )