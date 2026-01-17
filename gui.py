import os
import streamlit as st
import torch
import joblib
import json
from baseline_model import CNN_LSTM_VQA
from hybrid_model import CNN_BERT_VQA
from predict import predict_lstm, predict_biobert, predict_vilt
from transformers import ViltForQuestionAnswering, BertTokenizer, ViltProcessor, ViltImageProcessor
from PIL import Image


# --- PAGE CONFIG ---
st.set_page_config(page_title="Medical VQA Assistant", layout="centered")

# ================================================ FUNCTIONS ================================================
@st.cache_resource # This ensures the model only loads once, not on every click
def load_resources():
    # Get the directory where resources are located
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RES_DIR = os.path.join(BASE_DIR, "resources")
    le_path = os.path.join(RES_DIR, "label_encoder.joblib")
    vocab_path = os.path.join(RES_DIR, "vocab.json")
    lstm_path = os.path.join(RES_DIR, "best_lstm.pth")
    bert_path = os.path.join(RES_DIR, "best_bert.pth")
    vilt_path = os.path.join(RES_DIR, "best_vilt.pth")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Label Encoder
    le = joblib.load(le_path)
    num_classes = len(le.classes_)
    
    # Load LSTM Vocabulary
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
        
    # Initialize and Load CNN-LSTM Model
    lstm_model = CNN_LSTM_VQA(vocab_size, num_classes, unfreeze_resnet=False).to(device)
    lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
    
    # Initialize and Load CNN-BERT Model
    bert_model = CNN_BERT_VQA(num_classes, unfreeze_resnet=False, unfreeze_bert=False).to(device)
    bert_model.load_state_dict(torch.load(bert_path, map_location=device))
    
    # Initialize and Load ViLT Model
    vilt_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa",
                                                      num_labels=num_classes,
                                                      ignore_mismatched_sizes=True,
                                                      problem_type="single_label_classification").to(device)
    vilt_model.load_state_dict(torch.load(vilt_path, map_location=device))
    
    # Tokenizers/Processors
    bert_tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    image_processor = ViltImageProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa", do_rescale=False)
    vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa", image_processor=image_processor)
    
    return lstm_model, bert_model, vilt_model, bert_tokenizer, vilt_processor, le, vocab, device


# =============================================== APPLICATION ===============================================
# To start the application, run this command in terminal: streamlit run gui.py

# Initialize variables
lstm_m, bert_m, vilt_m, b_tok, v_proc, le, vocab, device = load_resources()

# --- GUI LAYOUT ---
st.title("🏥 Medical VQA System")
st.write("Upload a radiology image and ask a clinical question.")

# Image Upload
uploaded_file = st.file_uploader("Choose a medical image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the file name and image
    st.info(f"File uploaded: {uploaded_file.name}")
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Radiology Scan", width=250)

    # Model Selection
    model_choice = st.selectbox(
        "Choose Model Architecture",
        ("ResNet18 + LSTM", "ResNet50 + BioBERT", "ViLT")
    )

    # Question Input (Limited to 32 words)
    question = st.text_input("Enter your question:", placeholder="e.g., Is there a fracture?")
    
    word_count = len(question.split())
    if word_count > 32:
        st.warning(f"Warning: Your question has {word_count} words. It will be truncated to 32.")

    # Submit Button and Prediction
    if st.button("Generate Answer"):
        if not question:
            st.error("Please enter a question first.")
        else:
            with st.spinner(f"Analyzing using {model_choice}..."):
                try:
                    if model_choice == "ResNet18 + LSTM":
                        result = predict_lstm(image, question, lstm_m, le, vocab, device)
                    
                    elif model_choice == "ResNet50 + BioBERT":
                        result = predict_biobert(image, question, bert_m, b_tok, le, device)
                    
                    elif model_choice == "ViLT":
                        result = predict_vilt(image, question, vilt_m, v_proc, le, device)

                    # Display Prediction Result
                    st.success("Analysis Complete!")
                    st.metric(label="Predicted Answer", value=result)
                    
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

else:
    st.write("Please upload an image to begin.")
    