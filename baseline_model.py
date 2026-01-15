import torch

def predict_lstm(img, question, model, tokenizer, label_encoder, device):
    model.eval()
    
    # 1. Image Pre-processing (Match your training transforms)
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(img).unsqueeze(0).to(device) # Add batch dimension

    # 2. Question Pre-processing
    # Assuming your tokenizer has an encode method
    q_tokens = tokenizer.encode(question) 
    q_tensor = torch.tensor(q_tokens).unsqueeze(0).to(device)
    q_length = torch.tensor([len(q_tokens)])

    # 3. Forward Pass
    with torch.no_grad():
        output = model(img_tensor, q_tensor, q_length)
        pred_idx = output.argmax(dim=1).item()

    # 4. Map index to Answer
    return label_encoder[pred_idx]