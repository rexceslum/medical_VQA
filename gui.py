import streamlit as st
import torch
import joblib
import json

@st.cache_resource # This ensures the model only loads once, not on every click
def load_resources():
    # Load Label Encoder
    le = joblib.load('label_encoder.joblib')
    
    # Load LSTM Vocabulary
    with open('vocab.json', 'r') as f:
        vocab = json.load(f)
        
    # Initialize and Load Model (Example for LSTM)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_LSTM_VQA(len(vocab), len(le.classes_))
    model.load_state_dict(torch.load('best_lstm.pth', map_location=device))
    model.eval()
    
    return model, vocab, le, device

def process_and_predict(image_file, question_text, _model, _vocab, _le, _device):
    # 1. Convert Streamlit UploadedFile to PIL Image
    from PIL import Image
    img = Image.open(image_file).convert('RGB')
    
    # 2. Call the prediction function we wrote earlier
    # Note: Ensure you have the 'predict_lstm' or 'predict_bert' function defined
    prediction = predict_lstm(img, question_text, _model, _vocab, _le.classes_, _device)
    
    return prediction