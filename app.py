import joblib
import os
import shutil

from pathlib import Path

import streamlit as st

import requests

from bs4 import BeautifulSoup

from PIL import Image, ImageOps, ImageEnhance

import pytesseract

from ddgs import DDGS

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity



MODEL_PATH = Path("models/overall_model.joblib")



def load_authorship_model():

    if MODEL_PATH.exists():

        return joblib.load(MODEL_PATH)

    return None



authorship_model = load_authorship_model()


def predict_authorship(text):

    if authorship_model is None:

        return None



    if not text or len(text.strip()) < 20:

        return None



    probabilities = authorship_model.predict_proba([text])[0]

    classes = authorship_model.classes_



    results = dict(zip(classes, probabilities))



    ai_percentage = results.get("ai", 0) * 100

    human_percentage = results.get("human", 0) * 100



    return {

        "ai": round(ai_percentage, 1),

        "human": round(human_percentage, 1)

    }

# --------------------------------------------------

# PAGE CONFIGURATION

# --------------------------------------------------



st.set_page_config(

    page_title="AI Authorship Detector",

    page_icon="🔍",

    layout="wide"

)



# Tesseract OCR location
# Tesseract OCR location (Windows and Streamlit Cloud)
import os
import shutil

if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
else:
    pytesseract.pytesseract.tesseract_cmd = (
        shutil.which("tesseract") or "/usr/bin/tesseract"
    )

# --------------------------------------------------

# SESSION STATE

# --------------------------------------------------



if "input_text" not in st.session_state:

    st.session_state.input_text = ""



if "last_loaded_text" not in st.session_state:

    st.session_state.last_loaded_text = ""



if "review_text" not in st.session_state:

    st.session_state.review_text = ""



# --------------------------------------------------

# HEADER

# --------------------------------------------------



st.title("🔍 AI Authorship Detector")



st.write(

    "Analyze text, algorithms, explanations and images "

    "for possible AI authorship and online similarities."

)



st.info(

    "AI authorship detection is an estimate, not proof."

)



# --------------------------------------------------

# INPUT TABS

# --------------------------------------------------



tab1, tab2, tab3 = st.tabs([

    "📝 Paste Text",

    "🖼️ Upload Image",

    "🌐 Website URL"

])



# --------------------------------------------------

# 1. PASTE TEXT

# --------------------------------------------------



with tab1:



    pasted_text = st.text_area(

        "Paste your text here:",

        height=200,

        placeholder="Paste your text or algorithm here..."

    )



    if st.button("Use This Text"):



        st.session_state.input_text = pasted_text



        st.success("Text added successfully!")



# --------------------------------------------------

# 2. UPLOAD IMAGE AND EXTRACT TEXT

# --------------------------------------------------



with tab2:



    uploaded_image = st.file_uploader(

        "Upload an image",

        type=["png", "jpg", "jpeg", "bmp"]

    )



    if uploaded_image:



        image = Image.open(uploaded_image).convert("RGB")



        st.image(

            image,

            caption="Uploaded image",

            width=450

        )



        if st.button("Extract Text From Image"):



            try:



                # Convert image to grayscale

                processed_image = ImageOps.grayscale(image)



                # Improve contrast

                processed_image = ImageOps.autocontrast(

                    processed_image

                )



                # Enlarge image

                width, height = processed_image.size



                processed_image = processed_image.resize(

                    (width * 2, height * 2)

                )



                # Enhance contrast

                processed_image = ImageEnhance.Contrast(

                    processed_image

                ).enhance(1.5)



                # Extract text

                extracted_text = pytesseract.image_to_string(

                    processed_image,

                    config="--oem 3 --psm 6"

                )



                st.session_state.input_text = extracted_text



                st.success("Text extracted successfully!")



            except Exception as error:



                st.error(f"Image extraction failed: {error}")



# --------------------------------------------------

# 3. FETCH WEBSITE TEXT

# --------------------------------------------------



with tab3:



    website_url = st.text_input(

        "Enter a webpage URL:",

        placeholder="https://example.com/article"

    )



    if st.button("Get Website Text"):



        try:



            if not website_url.startswith(

                ("https://", "http://")

            ):



                st.error("Please enter a valid URL.")



            else:



                response = requests.get(

                    website_url,

                    timeout=15,

                    headers={

                        "User-Agent": "Mozilla/5.0"

                    }

                )



                response.raise_for_status()



                soup = BeautifulSoup(

                    response.text,

                    "html.parser"

                )



                for element in soup(

                    [

                        "script",

                        "style",

                        "nav",

                        "footer",

                        "header"

                    ]

                ):



                    element.decompose()



                website_text = soup.get_text(

                    separator=" ",

                    strip=True

                )



                st.session_state.input_text = website_text



                st.success("Website text retrieved!")



        except requests.RequestException as error:



            st.error(f"Website error: {error}")



# --------------------------------------------------

# REVIEW TEXT BEFORE ANALYSIS

# --------------------------------------------------



st.divider()



st.subheader("📝 Review Text Before Analysis")



current_text = st.session_state.input_text



# Update editable text when a new source is loaded

if current_text != st.session_state.last_loaded_text:



    st.session_state.review_text = current_text



    st.session_state.last_loaded_text = current_text



if current_text:



    analysis_text = st.text_area(

        "Check and correct extracted text here:",

        key="review_text",

        height=250

    )



    col1, col2 = st.columns(2)



    col1.metric(

        "Total Words",

        len(analysis_text.split())

    )



    col2.metric(

        "Total Characters",

        len(analysis_text)

    )



else:



    analysis_text = ""



    st.warning(

        "Paste text, upload an image, or enter a website URL."

    )



# --------------------------------------------------

# ONLINE SIMILARITY SEARCH

# --------------------------------------------------



st.divider()



st.subheader("🌐 Online Similarity Search")



st.write(

    "Search publicly accessible webpages for similar text."

)



if st.button(

    "🔎 Search for Online Similarities",

    disabled=not analysis_text.strip()

):



    words = analysis_text.split()



    if len(words) < 8:



        st.warning(

            "Please provide at least 8 words for searching."

        )



    else:



        # Use a shorter search phrase

        query = " ".join(words[:15])



        try:



            with st.spinner("Searching the internet..."):



                results = list(

                    DDGS().text(

                        query,

                        max_results=5,

                        backend="yandex"

                    )

                )



            if results:



                st.success(

                    f"Found {len(results)} search results."

                )



                for index, result in enumerate(results, 1):



                    title = result.get(

                        "title",

                        "Untitled webpage"

                    )



                    body = result.get(

                        "body",

                        ""

                    )



                    link = result.get(

                        "href",

                        ""

                    )



                    st.markdown(

                        f"### {index}. {title}"

                    )



                    st.write(body)



                    # Compare submitted text with search snippet

                    if body.strip():



                        try:



                            vectorizer = TfidfVectorizer()



                            vectors = vectorizer.fit_transform(

                                [

                                    analysis_text,

                                    body

                                ]

                            )



                            similarity = cosine_similarity(

                                vectors[0:1],

                                vectors[1:2]

                            )[0][0] * 100



                            st.metric(

                                "Similarity with search snippet",

                                f"{similarity:.1f}%"

                            )



                        except ValueError:



                            st.caption(

                                "Similarity could not be calculated."

                            )



                    if link:



                        st.markdown(

                            f"[Open webpage]({link})"

                        )



                    st.divider()



            else:



                st.info(

                    "No matching search results found."

                )



        except Exception as error:



            st.error(

                f"Online search failed: {error}"

            )



            st.info(

                "The free search service may be temporarily unavailable."

            )



# --------------------------------------------------

# OVERALL AI AUTHORSHIP ESTIMATE

# --------------------------------------------------



st.divider()



st.subheader("🧠 Overall Authorship Estimate")



if authorship_model is None:



    st.warning("No trained model found. Train the model first.")



elif analysis_text.strip():



    prediction = predict_authorship(analysis_text)



    if prediction is not None:



        col1, col2 = st.columns(2)



        col1.metric(

            "Human estimate",

            f"{prediction['human']}%"

        )



        col2.metric(

            "AI estimate",

            f"{prediction['ai']}%"

        )



        st.caption(

            "Experimental model scores only. "

            "These are not proof of authorship and "

            "may be inaccurate."

        )



    else:

        st.info("Enter at least 20 characters to analyze.")



else:

    st.info("Enter text above to see an authorship estimate.")



# --------------------------------------------------

# DETECTION STATUS

# --------------------------------------------------



st.divider()



st.subheader("📊 Detection Status")



st.write("🟢 Text input: Available")



st.write("🟢 Image OCR: Available")



st.write("🟢 Website text extraction: Available")



st.write("🟢 Free online search: Connected")

if authorship_model is not None:

    st.write("🟢 Overall authorship model: Loaded")

else:

    st.write("🔴 Overall authorship model: Not available")



st.caption(

    "Online similarity is not proof of AI authorship. "

    "Search snippets are only partial representations of webpages. "

    "The displayed similarity score is not a plagiarism percentage."

)



# --------------------------------------------------
