import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import config
from models import EncoderCNN, DecoderWithAttention
from utils import visualize_attention

def generate_caption(encoder, decoder, vocab, image_path, device, max_len=config.MAX_LEN):
    encoder.eval()
    decoder.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    image = Image.open(image_path).convert("RGB")
    tensor_img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(tensor_img)
        h, c = decoder.init_hidden_state(features)

        word = torch.tensor([vocab.stoi["<START>"]]).to(device)
        embeddings = decoder.embedding(word)

        result_caption = []
        attention_weights = []

        for t in range(max_len):
            attention_weighted_encoding, alpha = decoder.attention(features, h)
            gate = decoder.sigmoid(decoder.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding

            lstm_input = torch.cat([embeddings, attention_weighted_encoding], dim=1)
            h, c = decoder.decode_step(lstm_input, (h, c))
            preds = decoder.fc(h)

            predicted_word_idx = preds.argmax(dim=1).item()
            predicted_word = vocab.itos.get(predicted_word_idx, "<UNK>")

            if predicted_word == "<END>":
                break

            result_caption.append(predicted_word)
            attention_weights.append(alpha[0].cpu())

            word = torch.tensor([predicted_word_idx]).to(device)
            embeddings = decoder.embedding(word)

    return result_caption, attention_weights

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate caption and attention heatmap for an image.")
    parser.add_argument("--image", type=str, default="./data/flickr8k/Images/sample.jpg", help="Path to input image")
    parser.add_argument("--model", type=str, default="./checkpoints/best_model.pth.tar", help="Path to model checkpoint")
    args = parser.parse_args()

    device = config.DEVICE

    if not os.path.exists(args.model):
        print(f"Model checkpoint not found at {args.model}. Please train the model first using train.py.")
    else:
        import vocabulary
        torch.serialization.add_safe_globals([vocabulary.Vocabulary])
        ckpt = torch.load(args.model, map_location=device, weights_only=False)
        vocab = ckpt['vocab']

        encoder = EncoderCNN().to(device)
        decoder = DecoderWithAttention(
            attention_dim=config.ATTENTION_DIM,
            embed_dim=config.EMBED_SIZE,
            decoder_dim=config.DECODER_DIM,
            vocab_size=len(vocab)
        ).to(device)

        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['state_dict'])

        if not os.path.exists(args.image):
            print(f"Image not found at {args.image}. Creating a sample image.")
            os.makedirs(os.path.dirname(args.image) if os.path.dirname(args.image) else '.', exist_ok=True)
            Image.new("RGB", (224, 224), color="blue").save(args.image)

        caption, attention = generate_caption(encoder, decoder, vocab, args.image, device)
        print("Generated Caption:", " ".join(caption))

        visualize_attention(args.image, caption, attention, vocab, filename="output_attention.png")
