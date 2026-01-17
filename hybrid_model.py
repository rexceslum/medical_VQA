import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import BertModel


class CNN_BERT_VQA(nn.Module):
    def __init__(self, num_classes, unfreeze_resnet=True, unfreeze_bert=True):
        super().__init__()
        # Image feature extractor
        self.cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1) # 2048 output features
        self.cnn.fc = nn.Identity()

        # Freeze CNN layers, only the final classifier and the img_norm layer will learn
        for p in self.cnn.parameters():
            p.requires_grad = False

        # Unfreeze the final block (layer4) that contains the most complex, high-level features
        if unfreeze_resnet:
            for p in self.cnn.layer4.parameters():
                p.requires_grad = True

        # Normalize to prevent the image modality from "overpowering" the text during the fusion step
        self.img_norm = nn.LayerNorm(2048)

        # BERT text encoder
        self.bert = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

        # Freeze BERT layers
        for p in self.bert.parameters():
            p.requires_grad = False

        # Freeze BERT pooler (used for the [CLS] token summary)
        for param in self.bert.pooler.parameters():
                param.requires_grad = False

        if unfreeze_bert:
            target_layer = self.bert.encoder.layer[11]

            for name, param in target_layer.named_parameters():
                # Unfreeze LayerNorm (very effective for domain shift)
                # Unfreeze all biases (helps shift the decision boundaries)
                if "LayerNorm" in name or "bias" in name:
                    param.requires_grad = True

        # BERT dropout
        self.text_dropout = nn.Dropout(0.3)

        # Fusion dropout
        self.fusion_dropout = nn.Dropout(0.6)

        # Final classifier 2048 (ResNet50) + 768 (BERT Pooler) = 2816
        self.classifier = nn.Linear(2048 + 768, num_classes)

    def forward(self, imgs, input_ids, attention_mask):
        img_feat = self.cnn(imgs)
        img_feat = torch.flatten(img_feat, 1)
        img_feat = self.img_norm(img_feat)

        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Mean pooling over valid tokens
        last_hidden = bert_out.last_hidden_state  # (B, T, 768)
        mask = attention_mask.unsqueeze(-1)       # (B, T, 1)

        text_feat = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
        text_feat = self.text_dropout(text_feat)

        combined = torch.cat([img_feat, text_feat], dim=1)
        combined = self.fusion_dropout(combined)
        return self.classifier(combined)