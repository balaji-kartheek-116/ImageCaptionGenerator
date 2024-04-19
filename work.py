import streamlit as st
import os
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
import torch.nn.functional as F
from PIL import Image
import pdfplumber
from gtts import gTTS  # Import Google Text-to-Speech library
import base64


# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message=".*Your .* pad_token_id .*")

# Load the pre-trained model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Function to predict caption from image
def predict_caption(image_paths):
    captions = []
    for image_path in image_paths:
        try:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")
            pixel_values = feature_extractor(images=[i_image], return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)
            attention_mask = F.pad(torch.ones(pixel_values.shape[:-1], dtype=torch.long, device=pixel_values.device),
                                   (0, pixel_values.shape[1] - 1), value=0)
            output_ids = model.generate(pixel_values, attention_mask=attention_mask, **gen_kwargs)
            pred = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            captions.append(pred)
        except Exception as e:
            print(f"Error processing image at path {image_path}: {e}")
            captions.append("This Image is not recognized, mention a good image")  # Append None for failed image
    return captions

# Function to extract text and images from PDF
def extract_text_and_images_from_pdf(pdf_path, output_folder):
    text = ""
    image_paths = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text += page.extract_text()
            text += "\n"  # Add a newline after each page

            images_list = page.images
            if images_list:
                for i, image_obj in enumerate(images_list, start=1):
                    try:
                        image_data = image_obj["stream"].get_data()
                        image_ext = "png" if image_data.startswith(b"\x89PNG") else "jpg"
                        image_path = os.path.join(output_folder, f"page_{page_number}_image_{i}.{image_ext}")
                        with open(image_path, "wb") as image_file:
                            image_file.write(image_data)
                        image_paths.append(image_path)
                    except Exception as e:
                        print(f"Error extracting image: {e}")

    return text.strip(), image_paths

# Function to save text as audio
def save_text_as_audio(text, audio_path):
    tts = gTTS(text=text, lang='en')
    tts.save(audio_path)

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )

# Streamlit app
def main():
    st.title("PDF Image Captioning")
    st.image("img.png")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        os.makedirs("uploaded_pdf", exist_ok=True)
        with open(os.path.join("uploaded_pdf", "uploaded_pdf.pdf"), "wb") as f:
            f.write(uploaded_file.getvalue())

        # Extract text and images from PDF
        extracted_text, image_paths = extract_text_and_images_from_pdf("uploaded_pdf/uploaded_pdf.pdf", "images")

        # Predict captions for the images
        captions = predict_caption(image_paths)

        # Display extracted text and image captions in desired format
        st.write("Extracted Text:")
        st.write(extracted_text)
        st.write("Extracted Image Captions:")
        for caption in captions:
            st.write(caption)

        # Save all the output text and captions as audio
        all_text = f"Extracted text from the PDF doc is: {extracted_text}\n\nNow the extracted captions for all the images are:\n" + "\n".join(captions)
        save_text_as_audio(all_text, "output_audio.mp3")

        # Display audio player
        #st.audio("output_audio.mp3", format="audio/mp3")
        autoplay_audio("output_audio.mp3")

if __name__ == "__main__":
    main()
