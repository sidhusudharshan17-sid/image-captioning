import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.transforms as transforms
from tqdm import tqdm

import config
from dataset import get_loader
from models import EncoderCNN, DecoderWithAttention
from utils import save_checkpoint

def train():
    device = config.DEVICE
    print(f"Using device: {device}")

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Ensure dataset directories exist or create dummy setup if testing
    if not os.path.exists(config.CAPTION_FILE):
        print(f"Warning: {config.CAPTION_FILE} not found. Please place Flickr8k dataset in {config.DATA_DIR}.")
        # Create a sample synthetic dataset for demonstration/testing if none exists
        os.makedirs(config.IMAGE_DIR, exist_ok=True)
        import pandas as pd
        from PIL import Image
        dummy_img_name = "sample.jpg"
        dummy_img_path = os.path.join(config.IMAGE_DIR, dummy_img_name)
        Image.new("RGB", (224, 224), color="blue").save(dummy_img_path)
        df = pd.DataFrame({
            "image": [dummy_img_name] * 5,
            "caption": ["a dog running in the park", "a blue image sample", "a dog playing outside", "an image of blue color", "a dog in the grass"]
        })
        os.makedirs(os.path.dirname(config.CAPTION_FILE), exist_ok=True)
        df.to_csv(config.CAPTION_FILE, index=False)
        print(f"Created sample dataset at {config.CAPTION_FILE}")

    # Load data
    dataloader, vocab = get_loader(
        root_dir=config.IMAGE_DIR,
        caption_file=config.CAPTION_FILE,
        transform=transform,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        freq_threshold=2
    )

    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # Initialize models
    encoder = EncoderCNN().to(device)
    decoder = DecoderWithAttention(
        attention_dim=config.ATTENTION_DIM,
        embed_dim=config.EMBED_SIZE,
        decoder_dim=config.DECODER_DIM,
        vocab_size=vocab_size
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])

    # Optimizers
    optimizer = optim.Adam(
        params=list(decoder.parameters()) + list(filter(lambda p: p.requires_grad, encoder.parameters())),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    best_loss = float('inf')

    for epoch in range(config.EPOCHS):
        encoder.train()
        decoder.train()
        epoch_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{config.EPOCHS}]")
        for idx, (imgs, captions) in enumerate(progress_bar):
            imgs = imgs.to(device)
            captions = captions.to(device)

            # Forward pass
            features = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas = decoder(features, captions[:, :-1], [len(c) for c in captions])

            # Targets (exclude <START> token)
            targets = captions[:, 1:]

            # Pack padded sequence to calculate loss efficiently
            scores_packed = pack_padded_sequence(scores, decode_lengths, batch_first=True, enforce_sorted=False).data
            targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=False).data

            # Calculate loss (including attention regularization if desired)
            loss = criterion(scores_packed, targets_packed)
            # Add doubly stochastic attention regularization
            loss += 1.0 * ((1. - alphas.sum(dim=1)) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] Completed. Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': decoder.state_dict(),
            'encoder': encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'vocab': vocab
        }
        save_checkpoint(checkpoint, os.path.join(config.SAVE_DIR, "latest_model.pth.tar"))

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(checkpoint, os.path.join(config.SAVE_DIR, "best_model.pth.tar"))
            print("=> Saved new best model checkpoint.")

if __name__ == "__main__":
    train()
