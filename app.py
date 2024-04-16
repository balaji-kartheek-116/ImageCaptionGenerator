import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import torch.nn.functional as F
import fitz
import os

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
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    attention_mask = F.pad(torch.ones(pixel_values.shape[:-1], dtype=torch.long, device=pixel_values.device), (0, pixel_values.shape[1] - 1), value=0)

    output_ids = model.generate(pixel_values, attention_mask=attention_mask, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

# Streamlit app
def main():
    st.title("PDF Image Captioning")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Create a folder to store the uploaded PDF file
        os.makedirs("uploaded_pdf", exist_ok=True)
        # Save the uploaded PDF file
        with open(os.path.join("uploaded_pdf", "uploaded_pdf.pdf"), "wb") as f:
            f.write(uploaded_file.getvalue())

        # Open the uploaded PDF file
        with fitz.open(os.path.join("uploaded_pdf", "uploaded_pdf.pdf")) as pdf_file:
            with st.spinner("Processing PDF..."):
                # Initialize text and image paths
                final_text = ""
                image_paths = []

                # Get the number of pages in PDF file
                page_nums = len(pdf_file)

                # Extract all images from each page and save them
                for page_num in range(page_nums):
                    page_content = pdf_file[page_num]
                    images_list = page_content.get_images()

                    # Save all the extracted images
                    for i, img in enumerate(images_list, start=1):
                        xref = img[0]
                        base_image = pdf_file.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        image_name = f"page_{page_num + 1}_image_{i}.{image_ext}"
                        image_path = os.path.join("images", image_name)
                        with open(image_path, "wb") as image_file:
                            image_file.write(image_bytes)
                        image_paths.append(image_path)

                    # Extract text from the page
                    text = page_content.get_text()
                    final_text += text + "\n"

                # Predict captions for the images
                captions = predict_caption(image_paths)

                # Display extracted text and image captions in desired format
                lines = final_text.split('\n')
                iter_lines = iter(lines)
                for line in iter_lines:
                    st.write(line)
                    if captions:
                        caption = captions.pop(0)
                        if caption:
                            st.write(f"{caption}")
                            try:
                                next_line = next(iter_lines)
                                st.write(next_line)
                            except StopIteration:
                                pass

if __name__ == "__main__":
    main()
