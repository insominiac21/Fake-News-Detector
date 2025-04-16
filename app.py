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
from dotenv import load_dotenv  # Import dotenv to load environment variables

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
#GOOGLE_API_KEY = os.getenv("api_key")
API_KEY = st.secrets["GOOGLE_API_KEY"]



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

def check_text_fact(text, api_key):
    """Use Google Fact Check API to verify the text."""
    endpoint = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query": text,
        "key": api_key
    }
    response = requests.get(endpoint, params=params)
    if response.status_code != 200:
        return "Error accessing Fact Check API", None
    
    data = response.json()
    
    # Debugging: Log the query and response
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
    """Predict if the given image is a deepfake using a pre-trained model."""
    response = requests.get(image_url, stream=True)
    if response.status_code != 200:
        return "Error fetching image"
    
    try:
        # Convert image to RGB format
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
st.title("Fake News Detector")
st.write("Enter a news article URL to check its authenticity.")

url = st.text_input("Enter News URL:")
apikey = API_KEY

if st.button("Check News"):
    if apikey and url:
        # Validate the URL
        if not validators.url(url) or not (url.startswith("http://") or url.startswith("https://")):
            st.error("Invalid URL. Please enter a valid HTTP or HTTPS URL.")
        else:
            st.write("Scraping the website...")
            text, images = scrape_website(url)
            
            text_flag = False  # Initialize text_flag with a default value
            
            if text:
                st.subheader("Extracted Text")
                st.write(text[:500] + "...")
                
                # Extract key sentences for fact-checking
                try:
                    nlp = spacy.load('en_core_web_sm')
                    doc = nlp(text)
                    key_claims = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PERSON', 'EVENT']]
                    key_sentences = key_claims[0] if key_claims else ' '.join(text.split('.')[:3])
                except:
                    key_sentences = ' '.join(text.split('.')[:3])  # Extract first 3 sentences
                
                st.write("Checking text authenticity...")
                text_result, review_details = check_text_fact(key_sentences, apikey)
                first_word = text_result.split()[0].rstrip('.') if text_result else ""  # Remove trailing period
                third_word = review_details.split()[2].rstrip('.') if review_details and len(review_details.split()) > 2 else ""  # Get third word
                
                if first_word == "No" or third_word == "No":
                    st.write("Could not run text review.")
                    st.write("Reason: The webpage has not been reviewed by Google Claim Review yet.")
                    text_flag = None
            
                elif first_word or third_word in {"Half true", "False", "Mostly", "Misrepresentation", "Pants", "Fake", "Incorrect", "Misleading", "No", "Out", "Unfounded", "Exaggerated", "Debunked"} or third_word in {"Half true", "False", "Mostly", "Misrepresentation", "Pants", "Fake", "Incorrect", "Misleading", "No", "Out", "Unfounded", "Exaggerated", "Debunked"}:
                    st.write("🚨 This news might be FAKE!")
                    st.write("Fact Check Result: ", text_result)
                    st.write("\n", review_details)
                    text_flag = True
                elif first_word == "Not":  # Not Transcript
                    st.write("Fact Check Result: Independent assessment provided")
                    st.write("\n", review_details)
                    text_flag = False    
                else:
                    st.write("Fact Check Result: ", text_result)
                    if review_details:
                        st.write("Supporting Evidence: ", review_details)
                    text_flag = False
            else:
                st.write("No text found on the page.")
                text_result = "Unknown"
                review_details = None
                text_flag = False  # Ensure text_flag is set even if no text is found
            
            if images:
                st.subheader("Extracted Images")
                model = load_model("deepfake_model.h5",compile=False)
                deepfake_results = {}
                
                for img_url in images[:3]:  # Limit to 3 images for performance
                    result = check_image_deepfake(img_url, model)
                    deepfake_results[img_url] = result
                    st.image(img_url, caption=result, use_container_width=True)
                
                # Calculate fake score
                fake_score = sum(1 for v in deepfake_results.values() if v == "Deepfake") / max(len(deepfake_results), 1)
            else:
                st.write("No images found.")
                fake_score = 0
            
            # Final Verdict Logic
            st.subheader("Final Verdict")
            st.write("Combining text and image analysis...")
            # Adjust confidence calculation to prioritize text_flag
            if text_flag is True:
                combined_confidence = max(fake_score, 0.7)  # At least 70% if text is flagged as fake
            elif text_flag is None:
                combined_confidence = fake_score  # Use only fake_score if no fact-check is available
            else:
                combined_confidence = fake_score * 0.5  # Reduce weight of fake_score if text is real
            
            # Display final verdict
            if text_flag is True and combined_confidence > 0.5:
                st.error(f"🚨 This news might be FAKE! Confidence: {combined_confidence * 100:.2f}%")
            elif text_flag is None and combined_confidence > 0.5:
                st.warning(f"⚠️ This news might be PARTIALLY FAKE. Confidence: {combined_confidence * 100:.2f}%")
            elif text_flag is False or combined_confidence <= 0.5:
                st.success(f"✅ This news appears REAL. Confidence: {(1 - combined_confidence) * 100:.2f}%")
    else:
        st.warning("Please enter a valid URL and API Key.")
