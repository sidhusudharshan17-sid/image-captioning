import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    torch.save(state, filename)

def load_checkpoint(filename, model, optimizer=None):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint not found: {filename}")
    import vocabulary
    torch.serialization.add_safe_globals([vocabulary.Vocabulary])
    checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint.get('epoch', 0)

def visualize_attention(image_path, result_caption, attention_weights, vocab, filename="attention_plot.png"):
    """
    Visualizes attention weights on the image for each word in the generated caption.
    """
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    plt.figure(figsize=(15, 10))
    num_words = len(result_caption)

    # Plot original image first
    plt.subplot(1, num_words + 1, 1)
    plt.imshow(image_np)
    plt.title("Original")
    plt.axis('off')

    # Plot attention over image for each word
    for t in range(num_words):
        if t >= len(attention_weights):
            break
        plt.subplot(1, num_words + 1, t + 2)
        plt.imshow(image_np)
        plt.axis('off')

        # Get attention weights for 14x14 grid
        att_map = attention_weights[t].reshape(14, 14).detach().cpu().numpy()
        att_img = Image.fromarray(att_map).resize((image.size[0], image.size[1]), Image.BICUBIC)

        plt.imshow(att_img, cmap='jet', alpha=0.6)
        plt.title(result_caption[t], fontsize=12)

    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Attention plot saved to {filename}")
