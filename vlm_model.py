import torch

def predict_vilt(img, question, model, processor, label_encoder, device):
    model.eval()
    
    # 1. Unified Pre-processing
    encoding = processor(img, question, return_tensors="pt").to(device)

    # 2. Forward Pass
    with torch.no_grad():
        outputs = model(**encoding)
        # ViLT from HuggingFace returns an object; we need the logits
        pred_idx = outputs.logits.argmax(dim=-1).item()

    return label_encoder[pred_idx]