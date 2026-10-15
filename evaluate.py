import torch
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from tqdm import tqdm
from models import EncoderCNN, DecoderWithAttention
from dataset import get_loader
import config
import torchvision.transforms as transforms
import os

def evaluate_model(encoder, decoder, dataloader, vocab, device):
    encoder.eval()
    decoder.eval()

    references = []
    hypotheses = []

    with torch.no_grad():
        for imgs, captions in tqdm(dataloader, desc="Evaluating"):
            imgs = imgs.to(device)
            captions = captions.to(device)

            features = encoder(imgs)
            batch_size = features.size(0)

            # Generate captions using greedy decoding
            h, c = decoder.init_hidden_state(features)

            # Start token
            word = torch.tensor([vocab.stoi["<START>"]] * batch_size).to(device)
            embeddings = decoder.embedding(word)

            decoded_batch = [[] for _ in range(batch_size)]
            finished = [False] * batch_size

            for t in range(config.MAX_LEN):
                attention_weighted_encoding, alpha = decoder.attention(features, h)
                gate = decoder.sigmoid(decoder.f_beta(h))
                attention_weighted_encoding = gate * attention_weighted_encoding

                lstm_input = torch.cat([embeddings, attention_weighted_encoding], dim=1)
                h, c = decoder.decode_step(lstm_input, (h, c))
                preds = decoder.fc(h)

                predicted_words = preds.argmax(dim=1)

                for i in range(batch_size):
                    if finished[i]:
                        continue
                    w_idx = predicted_words[i].item()
                    w_word = vocab.itos.get(w_idx, "<UNK>")

                    if w_word == "<END>" or t == config.MAX_LEN - 1:
                        finished[i] = True
                    else:
                        decoded_batch[i].append(w_word)

                embeddings = decoder.embedding(predicted_words)

            # Collect references and hypotheses for BLEU
            for i in range(batch_size):
                # Reference caption (exclude <START>, <END>, <PAD>)
                ref = [vocab.itos.get(token.item(), "<UNK>") for token in captions[i] if token.item() not in [vocab.stoi["<PAD>"], vocab.stoi["<START>"], vocab.stoi["<END>"]]]
                references.append([ref])
                hypotheses.append(decoded_batch[i])

    # Compute BLEU-4 score
    chencherry = SmoothingFunction().method1
    bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0), smoothing_function=chencherry)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry)

    print(f"\nEvaluation Results:")
    print(f"BLEU-1: {bleu1 * 100:.2f}%")
    print(f"BLEU-2: {bleu2 * 100:.2f}%")
    print(f"BLEU-4: {bleu4 * 100:.2f}%")

    return bleu4

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    # Quick test runner if dataset exists
    if os.path.exists(config.CAPTION_FILE):
        loader, vocab = get_loader(config.IMAGE_DIR, config.CAPTION_FILE, transform=transform, batch_size=32, shuffle=False)
        encoder = EncoderCNN().to(config.DEVICE)
        decoder = DecoderWithAttention(config.ATTENTION_DIM, config.EMBED_SIZE, config.DECODER_DIM, len(vocab)).to(config.DEVICE)

        ckpt_path = os.path.join(config.SAVE_DIR, "best_model.pth.tar")
        if os.path.exists(ckpt_path):
            import vocabulary
            torch.serialization.add_safe_globals([vocabulary.Vocabulary])
            ckpt = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
            vocab = ckpt['vocab']
            encoder = EncoderCNN().to(config.DEVICE)
            decoder = DecoderWithAttention(config.ATTENTION_DIM, config.EMBED_SIZE, config.DECODER_DIM, len(vocab)).to(config.DEVICE)
            encoder.load_state_dict(ckpt['encoder'])
            decoder.load_state_dict(ckpt['state_dict'])
            evaluate_model(encoder, decoder, loader, vocab, config.DEVICE)
        else:
            print("Checkpoint not found. Train the model first.")
    else:
        print(f"Caption file {config.CAPTION_FILE} not found.")
