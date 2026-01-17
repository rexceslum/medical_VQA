import torch
from PIL import Image
from torchvision import transforms


def tokenize_question(text, vocab, max_len=32):
    # Map unknown words to 0 (<unk>) and clip indices to vocab_size-1
    vocab_size = len(vocab)
    tokens = [vocab.get(w, vocab["<unk>"]) for w in text.lower().split()]
    tokens = [min(t, vocab_size-1) for t in tokens]  # <-- clamp to avoid out-of-range
    length = min(len(tokens), max_len)
    if len(tokens) < max_len:
        tokens += [vocab["<pad>"]] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return torch.tensor(tokens, dtype=torch.long), length

def apply_square_padding(img):
    old_w, old_h = img.size
    max_dim = max(old_w, old_h)
    
    # Create black square canvas
    new_image = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
    
    # Center the original image
    paste_x = (max_dim - old_w) // 2
    paste_y = (max_dim - old_h) // 2
    new_image.paste(img, (paste_x, paste_y))
    
    # Resize to the model's expected input size
    return new_image.resize((224, 224))

def predict_lstm(img, question, model, le, vocab, device):
    # Image pre-processing
    img_padded = apply_square_padding(img)
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img_padded).unsqueeze(0).to(device) # Add batch dimension
    
    # Text preprocessing
    tokens, length = tokenize_question(question, vocab)
    q_tensor = tokens.unsqueeze(0).to(device)
    q_length = torch.tensor([length], dtype=torch.long).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(img_tensor, q_tensor, q_length)
        pred_idx = output.argmax(dim=1).item()

    # Map index to answer
    return le.inverse_transform([pred_idx])[0]

def predict_biobert(img, question, model, tokenizer, le, device):
    # Image pre-processing
    img_padded = apply_square_padding(img)
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img_padded).unsqueeze(0).to(device)

    # BioBERT tokenization for text preprocessing
    enc = tokenizer(
        [question], 
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )

    # Predict
    model.eval()
    with torch.no_grad():
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        output = model(img_tensor, input_ids, attention_mask)
        pred_idx = output.argmax(dim=1).item()

    return le.inverse_transform([pred_idx])[0]

def predict_vilt(img, question, model, processor, le, device):
    # Unified image and text preprocessing
    print(f"DEBUG: Processor type is {type(processor)}")
    img_padded = apply_square_padding(img)
    inputs = processor(
        images=img_padded,
        text=question,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
        # do_rescale=False
    ).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        pred_idx = outputs.logits.argmax(dim=-1).item()

    return le.inverse_transform([pred_idx])[0]