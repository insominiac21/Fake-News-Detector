# Fake News Detection Project

This web application, Fake News Detector, allows users to input a news article URL to verify its authenticity by analyzing both text and images. It scrapes the webpage for text content and images, uses the Google Fact Check API to validate key claims, and employs a deepfake detection model to analyze images. The app combines these results to provide a final verdict with a confidence score, indicating whether the news is real, partially fake, or fake. Built with Streamlit, it integrates libraries like BeautifulSoup, TensorFlow, and Spacy for seamless functionality

## Files
- `app.py`: The main application script.
- `requirements.txt`: Contains the list of required Python modules.

## How to Run

1. **Install Dependencies**  
    Run the following command to install the required modules:
    ```bash
    pip install -r requirements.txt
    ```

2. **Add Your API Key**  
    Update the `app.py` file with your API key. Replace `YOUR_API_KEY` with your actual key:
    ```python
    api_key = "YOUR_API_KEY"
    ```

3. **Run the Application**  
    Execute the script using:
    ```bash
    streamlit run app.py
    ```

## Requirements
The `requirements.txt` file includes the following modules:
- `streamlit`
- `requests`
- `numpy`
- `beautifulsoup4`
- `tensorflow`
- `pillow`
- `validators`
- `spacy`

Ensure these modules are installed before running the application.
