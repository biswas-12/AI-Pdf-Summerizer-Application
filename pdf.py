import os
import streamlit as st
import PyPDF2
from PIL import Image
import io
from google.cloud import vision
from pdf2image import convert_from_path

import vertexai
from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel
)

# Set up Google Cloud Vision Client
vision_client = vision.ImageAnnotatorClient()

# Set the page layout first
st.set_page_config(layout="wide")

# Initialize Vertex AI
PROJECT_ID = os.getenv("cloud-chef-439008")  # Replace with your actual env var name
LOCATION = os.getenv("us-east1")  # Default to 'us-east1' if not set
vertexai.init(project=PROJECT_ID, location=LOCATION)

@st.cache_resource
def load_model():
    return GenerativeModel("gemini-1.5-pro-002")  # Replace with a valid model

def create_prompt(text, option):
    """Creates a prompt based on the selected option."""
    if option == "Summary":
        return f"""
            You are a highly knowledgeable subject matter expert, acting as a dedicated tutor. Your goal is to create a comprehensive yet easily understandable 
            summary of the provided book content for a student. Analyze the following input, which may consist of text and images. If images are present, first perform 
            Optical Character Recognition (OCR) to extract the text.

            Your task is to create a detailed yet easy-to-understand summary of the book. Start by providing a brief overview of its main idea and key themes. If the book is 
            divided into chapters, summarize each chapter clearly, focusing on its main events, key arguments, and important concepts. Include the most important points in concise 
            bullet points for quick reference. Simplify complex ideas using plain language, ensuring they are easy for beginners to understand, and avoid technical jargon. Finally, 
            ensure the summary covers all essential aspects of the book, providing enough detail for thorough understanding, while keeping it digestible.

            Input Content:
            {text}

            BULLET POINT SUMMARY:
        """
    elif option == "Short Notes":
        return f"""
            Assume the role of a highly knowledgeable expert with deep understanding of the subject matter in the provided text. Your task is to create detailed, high-quality study notes 
            that cover all essential concepts, explanations, and key details from the text. The notes should be long enough to provide comprehensive coverage, ensuring every important point 
            is captured, but still clear and concise enough for easy understanding. Use your expertise to break down complex ideas into digestible segments, and structure the notes logically 
            with clear headings, bullet points, and other formatting techniques to improve readability. The goal is to provide the most effective study notes possible for someone preparing for 
            exams—thorough, well-organized, and accessible for quick revision.

            Please generate the notes after I provide the text.

            ```{text}```

            SHORT NOTES:
        """

    elif option == "Possible QnA":
        return f"""
            You are a knowledgeable tutor. Generate a set quizzes questions based on the provided content, making it simple and easy to understand. If the content includes both text and 
            images, first perform OCR to extract the text. Start by providing questions that focus on the main ideas and key themes of the content. Then, generate clear and concise answers, 
            simplifying any complex concepts using plain language. Ensure that the questions cover important concepts, and details. The goal is to help the reader 
            review and understand the material easily.

            Please generate the quizzes questions after I provide the text.

                ```{text}```

                QUESTIONS AND ANSWERS:
        """                    
    else:
        return "Invalid option selected."

def extract_text_from_image(image_file):
    """Extracts text from an image using Google Cloud Vision."""
    try:
        with io.BytesIO(image_file.read()) as image_bytes:
            image = vision.Image(content=image_bytes.getvalue())
            response = vision_client.text_detection(image=image)
            texts = response.text_annotations
            if texts:
                return texts[0].description
            else:
                st.warning("No text found in the uploaded image.")
                return ""
    except Exception as e:
        st.error(f"Error extracting text from image: {e}")
        return ""

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    full_text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        full_text += page.extract_text()
    return full_text

def extract_text_from_pdf_images(pdf_file):
    """Extract text from PDF pages with images using OCR."""
    full_text = ""
    images = convert_from_path(pdf_file)  # Converts PDF pages to images
    for image in images:
        with io.BytesIO() as img_byte:
            image.save(img_byte, format="PNG")
            img_byte.seek(0)
            full_text += extract_text_from_image(img_byte)
    return full_text

def get_text_response(model, prompt, config):
    responses = model.generate_content(prompt, generation_config=config, stream=True)
    return " ".join(response.text for response in responses if response.text)

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Streamlit app layout
st.markdown('<div class="header"><h1>AI Study Buddy</h1></div>', unsafe_allow_html=True)

col1, separator, col2 = st.columns([1, 0.02, 4], gap="small")

with col1:
    st.markdown('<div class="left-col">', unsafe_allow_html=True)
    st.subheader("About the App")
    st.write(
        "This app allows you to upload files (PDFs, TXT, or images), count the total number of words, and generate outputs such as a summary, short notes, or Q&A based on your selection."
    )
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "pdf"])
    option = st.radio("Choose Output Type", options=["Summary", "Short Notes"])
    generate = st.button("Generate")
    st.markdown('</div>', unsafe_allow_html=True)

with separator:
    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="right-col">', unsafe_allow_html=True)
    if uploaded_file:
        # Extract text from uploaded file
        full_text = ""
        if uploaded_file.type == "application/pdf":
            # Check if the PDF contains images or text
            full_text = extract_text_from_pdf(uploaded_file) or extract_text_from_pdf_images(uploaded_file)
        elif uploaded_file.type == "text/plain":
            full_text = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
            full_text = extract_text_from_image(uploaded_file)

        st.write(f"Total Words: {len(full_text.split())}")

        if generate:
            # Limit text length to fit token limit
            text_snippet = full_text[:30000]
            prompt = create_prompt(text_snippet, option)
            model = load_model()
            config = {"max_output_tokens": 8000}

            with st.spinner("Generating output..."):
                response = get_text_response(model, prompt, config)
                if response:
                    st.subheader(f"{option} Output:")
                    st.write(response)
    st.markdown('</div>', unsafe_allow_html=True)
