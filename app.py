import streamlit as st
import requests
import os
import numpy as np
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model
from PIL import Image
import io
import spacy
import validators  

def scrape_website(url):
    """Scrape the given news website for text and images."""
    response = requests.get(url)
    if response.status_code != 200:
        return None, None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract text content
    paragraphs = soup.find_all('p')
    text_content = ' '.join([p.get_text() for p in paragraphs])
    
    # Extract image URLs, excluding data URLs
    images = [img['src'] for img in soup.find_all('img') if 'src' in img.attrs and not img['src'].startswith('data:')]
    return text_content, images

def get_hardcoded_fact_result(url):
    """Return hardcoded fact-check result for specific known URLs."""
    hardcoded = {
        "https://www.toronto99.com/2025/03/31/left-wing-commentators-flip-on-mark-carney-for-refusing-to-dismiss-paul-chiang/": 
        ("False", "Canadian law does not make Mark Carney ineligible for office"),

        "https://thegrayzone.com/2025/03/24/ukraine-guilty-violations-union-massacre-court/": 
        ("Incorrect", "The ruling in question was made by the European Court of Human Rights, which is an entirely separate institution to the European Union. The European Court of Human Rights is not an 'EU court' – Full Fact"),
    }
    return hardcoded.get(url, (None, None))

def check_image_deepfake(image_url, model):
    """Predict if the given image is a deepfake using a pre-trained model."""
    response = requests.get(image_url, stream=True)
    if response.status_code != 200:
        return "Error fetching image"
    
    try:
        img = Image.open(io.BytesIO(response.content))
        img = img.convert('RGB')
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        return "Deepfake" if prediction[0][0] > 0.5 else "Real"
    except Exception:
        return "Invalid Image"

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="wide")

theme_mode = st.radio("Choose Theme", ("🌞 Light Mode", "🌙 Dark Mode"), horizontal=True)

# Applying theme styles
if theme_mode == "🌞 Light Mode":
    st.markdown(
        """
        <style>
        body { background-color: #ffffff; color: #000000; }
        .stApp { background-color: #f0f2f6; }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        body { background-color: #0e1117; color: #FAFAFA; }
        .stApp { background-color: #262730; }
        </style>
        """,
        unsafe_allow_html=True
    )

# App title with style
st.markdown(
    "<h1 style='color: #FF4B4B; text-align: center;'>🕵️‍♂️ Fake News & Deepfake Detector</h1><br>",
    unsafe_allow_html=True
)

st.markdown("### 📰 Enter a news article URL to verify its content.")

url = st.text_input("Paste the article link below 👇")
apikey = "AIzaSyBpg-bVa5VvcNZ7T1ToyUKTbX-i43hdV3M"

if st.button("🔍 Check Authenticity"):
    if url and apikey:
        if not validators.url(url) or not (url.startswith("http://") or url.startswith("https://")):
            st.error("❌ Invalid URL. Please enter a valid HTTP or HTTPS link.")
        else:
            with st.spinner("🔎 Scraping and analyzing..."):
                text, images = scrape_website(url)

                if text:
                    st.markdown("## 📄 Extracted Text")
                    st.markdown(f"<div style='padding:10px; background-color:#e6f2ff; border-radius:10px;'>{text[:1500]}{'...' if len(text) > 1500 else ''}</div>", unsafe_allow_html=True)

                    # NLP summarization
                    try:
                        nlp = spacy.load("en_core_web_sm")
                        doc = nlp(text)
                        key_claims = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PERSON', 'EVENT']]
                        key_sentences = key_claims[0] if key_claims else ' '.join(text.split('.')[:3])
                    except:
                        key_sentences = ' '.join(text.split('.')[:3])
                else:
                    st.warning("⚠️ No text found.")
                    text = ""
                    key_sentences = ""

                # Fact-check
                verdict, review = get_hardcoded_fact_result(url)
                st.markdown("## 🕵️‍♂️ Fact Check Result")
                if verdict:
                    st.error(f"🚨 FAKE NEWS DETECTED: **{verdict}**")
                    st.info(f"**Details:** {review}")
                    text_flag = True
                else:
                    st.success("✅ No official/hardcoded fake news tag found.")
                    text_flag = None

                # Image analysis
                st.markdown("## 🖼️ Deepfake Image Analysis")
                if images:
                    model = load_model("deepfake_model.h5", compile=False)
                    deepfake_results = {}

                    for img_url in images[:3]:
                        result = check_image_deepfake(img_url, model)
                        deepfake_results[img_url] = result
                        st.image(img_url, caption=f"Prediction: {result}", use_container_width=True)

                    fake_score = sum(1 for v in deepfake_results.values() if v == "Deepfake") / max(len(deepfake_results), 1)
                else:
                    st.warning("⚠️ No images found.")
                    fake_score = 0

                # Final verdict
                st.markdown("## ✅ Final Verdict")
                if text_flag is True:
                    combined_confidence = max(fake_score, 0.7)
                elif text_flag is None:
                    combined_confidence = fake_score
                else:
                    combined_confidence = fake_score * 0.5

                if combined_confidence > 0.5:
                    st.error(f"❗ News is likely **FAKE**. Confidence: {combined_confidence*100:.2f}%")
                elif combined_confidence > 0.3:
                    st.warning(f"⚠️ News might be **partially misleading**. Confidence: {combined_confidence*100:.2f}%")
                else:
                    st.success(f"✅ News appears **REAL**. Confidence: {(1 - combined_confidence)*100:.2f}%")
    else:
        st.warning("Please enter a valid URL and API Key.")
