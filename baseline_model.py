import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


class CNN_LSTM_VQA(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=256, hidden_dim=128, unfreeze_resnet=True):
        super().__init__()
        # Image feature extractor
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) # 512 output features
        self.cnn.fc = nn.Identity()

        # Freeze CNN layers, only the final classifier and the img_norm layer will learn
        for p in self.cnn.parameters():
            p.requires_grad = False

        # Unfreeze the final block (layer4) that contains the most complex, high-level features
        if unfreeze_resnet:
            for p in self.cnn.layer4.parameters():
                p.requires_grad = True

        # Normalize to prevent the image modality from "overpowering" the text during the fusion step
        self.img_norm = nn.LayerNorm(512)

        # Question encoder
        self.embed = nn.Embedding(vocab_size, 256, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        # LSTM dropout prevents overfitting on tiny dataset
        # forces the model to understand the question even if it "misses" a few words
        self.q_dropout = nn.Dropout(0.3)

        # Fusion dropout prevents the final Linear layer from relying on a single specific "pattern" of image+text
        self.fusion_dropout = nn.Dropout(0.6)

        # Final classifier 512 (CNN) + 128 (LSTM) = 1024
        self.fc = nn.Linear(512 + hidden_dim, num_classes)

    def forward(self, img, q_tokens, q_lengths):
        img_feat = self.cnn(img)
        # Flatten (batch_size, 512, 1, 1) to (batch_size, 512)
        img_feat = torch.flatten(img_feat, 1)
        # Normalize
        img_feat = self.img_norm(img_feat)

        q_emb = self.embed(q_tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            q_emb, q_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        # We take the last layer hidden state of the LSTM
        _, (h, _) = self.lstm(packed)
        q_feat = h[-1]
        q_feat = self.q_dropout(q_feat)

        # Fuse the image features and text features
        combined = torch.cat([img_feat, q_feat], dim=1)
        combined = self.fusion_dropout(combined)
        return self.fc(combined)