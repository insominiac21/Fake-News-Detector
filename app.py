import streamlit as st
import requests
import os
import numpy as np
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64
import spacy
import validators

def scrape_website(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None, None
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text_content = ' '.join([p.get_text() for p in paragraphs])
    images = [img['src'] for img in soup.find_all('img') if 'src' in img.attrs and not img['src'].startswith('data:')]
    return text_content, images

def check_text_fact(text, api_key):
    endpoint = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query": text,
        "key": api_key
    }
    response = requests.get(endpoint, params=params)
    if response.status_code != 200:
        return "Error accessing Fact Check API", None
    data = response.json()
    print("Query Sent to API:", text)
    print("API Response:", data)
    if 'claims' in data and len(data['claims']) > 0:
        claim = data['claims'][0]
        claim_review = claim.get('claimReview', [{}])[0]
        textual_rating = claim_review.get('textualRating', 'Unknown')
        review_text = claim_review.get('title', 'No additional details available')
        return textual_rating, review_text
    return "No fact-check available", None

def check_image_deepfake(image_url, model):
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

def analyze_url(url, api_key):
    result = {
        "text_result": None,
        "review_details": None,
        "text_flag": None,
        "deepfake_results": {},
        "combined_confidence": 0.0
    }
    text, images = scrape_website(url)
    if text:
        try:
            nlp = spacy.load('en_core_web_sm')
            doc = nlp(text)
            key_claims = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PERSON', 'EVENT']]
            key_sentences = key_claims[0] if key_claims else ' '.join(text.split('.')[:3])
        except:
            key_sentences = ' '.join(text.split('.')[:3])
        text_result, review_details = check_text_fact(key_sentences, api_key)
        result["text_result"] = text_result
        result["review_details"] = review_details

        first_word = text_result.split()[0].rstrip('.') if text_result else ""
        third_word = review_details.split()[2].rstrip('.') if review_details and len(review_details.split()) > 2 else ""

        if first_word == "No" or third_word == "No":
            result["text_flag"] = None
        elif first_word or third_word in {"Half true", "False", "Mostly", "Misrepresentation", "Pants", "Fake", "Incorrect", "Misleading", "No", "Out", "Unfounded", "Exaggerated", "Debunked"}:
            result["text_flag"] = True
        elif first_word == "Not":
            result["text_flag"] = False
        else:
            result["text_flag"] = False
    else:
        result["text_flag"] = False

    if images:
        model = load_model("deepfake_model.h5", compile=False)
        deepfake_results = {}
        for img_url in images[:3]:
            deepfake_results[img_url] = check_image_deepfake(img_url, model)
        result["deepfake_results"] = deepfake_results
        fake_score = sum(1 for v in deepfake_results.values() if v == "Deepfake") / max(len(deepfake_results), 1)
    else:
        fake_score = 0

    if result["text_flag"] is True:
        result["combined_confidence"] = max(fake_score, 0.7)
    elif result["text_flag"] is None:
        result["combined_confidence"] = fake_score
    else:
        result["combined_confidence"] = fake_score * 0.5

    return result

# Streamlit UI
st.title("Fake News Detector")
st.write("Enter a news article URL to check its authenticity.")

url = st.text_input("Enter News URL:")
apikey = "AIzaSyBpg-bVa5VvcNZ7T1ToyUKTbX-i43hdV3M"

if st.button("Check News"):
    if url and apikey:
        if not validators.url(url) or not (url.startswith("http://") or url.startswith("https://")):
            st.error("Invalid URL. Please enter a valid HTTP or HTTPS URL.")
        else:
            st.write("Scraping and analyzing the website...")
            result = analyze_url(url, apikey)

            if result["text_result"]:
                st.subheader("Fact Check Result")
                st.write("Result:", result["text_result"])
                if result["review_details"]:
                    st.write("Supporting Evidence:", result["review_details"])
            else:
                st.write("No fact-check result available.")

            if result["deepfake_results"]:
                st.subheader("Image Analysis")
                for img_url, status in result["deepfake_results"].items():
                    st.image(img_url, caption=status, use_container_width=True)
            else:
                st.write("No images found or processed.")

            st.subheader("Final Verdict")
            if result["text_flag"] is True and result["combined_confidence"] > 0.5:
                st.error(f"🚨 This news might be FAKE! Confidence: {result['combined_confidence'] * 100:.2f}%")
            elif result["text_flag"] is None and result["combined_confidence"] > 0.5:
                st.warning(f"⚠️ This news might be PARTIALLY FAKE. Confidence: {result['combined_confidence'] * 100:.2f}%")
            else:
                st.success(f"✅ This news appears REAL. Confidence: {(1 - result['combined_confidence']) * 100:.2f}%")
    else:
        st.warning("Please enter a valid URL and ensure API key is available in Streamlit secrets.")
