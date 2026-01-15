import torch

def predict_bert(img, question, model, tokenizer, label_encoder, device):
    model.eval()
    
    # 1. Image Pre-processing
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    # 2. Text Pre-processing (BioBERT specific)
    inputs = tokenizer(
        question, 
        return_tensors="pt", 
        padding="max_length", 
        truncation=True, 
        max_length=128
    ).to(device)

    # 3. Forward Pass
    with torch.no_grad():
        output = model(img_tensor, inputs['input_ids'], inputs['attention_mask'])
        pred_idx = output.argmax(dim=1).item()

    return label_encoder[pred_idx]