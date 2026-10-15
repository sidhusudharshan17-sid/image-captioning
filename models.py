import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(EncoderCNN, self).__init__()
        self.enc_image_size = encoded_image_size

        # Load pretrained ResNet101
        try:
            weights = models.ResNet101_Weights.DEFAULT
            resnet = models.resnet101(weights=weights)
        except Exception:
            resnet = models.resnet101(pretrained=True)

        # Remove linear and pool layers (keep up to conv5_3 feature maps)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Adaptive pooling to ensure fixed spatial dimensions (e.g., 14x14)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()

    def forward(self, images):
        """
        Input: images tensor of shape (batch_size, 3, image_size, image_size)
        Output: feature maps tensor of shape (batch_size, num_pixels, encoder_dim)
                where num_pixels = encoded_image_size * encoded_image_size (e.g. 196)
                      encoder_dim = 2048
        """
        out = self.resnet(images)                  # (batch_size, 2048, H/32, W/32)
        out = self.adaptive_pool(out)             # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)             # (batch_size, encoded_image_size, encoded_image_size, 2048)
        out = out.view(out.size(0), -1, out.size(-1)) # (batch_size, num_pixels, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False
        # Fine-tune last convolutional block (layer4)
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    Bahdanau (Additive) Attention mechanism.
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)     # Linear layer to transform encoder output
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)     # Linear layer to transform decoder hidden state
        self.full_att = nn.Linear(attention_dim, 1)                  # Linear layer to calculate scores to be softmaxed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)                             # Softmax to get attention weights over image regions

    def forward(self, encoder_out, decoder_hidden):
        """
        encoder_out: (batch_size, num_pixels, encoder_dim)
        decoder_hidden: (batch_size, decoder_dim)
        Returns: context vector (batch_size, encoder_dim), attention weights (batch_size, num_pixels)
        """
        att1 = self.encoder_att(encoder_out)                         # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)         # (batch_size, 1, attention_dim)
        combined = self.relu(att1 + att2)                            # (batch_size, num_pixels, attention_dim)
        score = self.full_att(combined).squeeze(2)                   # (batch_size, num_pixels)
        alpha = self.softmax(score)                                  # (batch_size, num_pixels)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)      # (batch_size, encoder_dim)

        return context, alpha


class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super(DecoderWithAttention, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # Attention network
        self.embedding = nn.Embedding(vocab_size, embed_dim)                 # Embedding layer
        self.dropout = nn.Dropout(p=dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim) # LSTM cell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)                    # Linear layer to find initial hidden state of LSTM
        self.init_c = nn.Linear(encoder_dim, decoder_dim)                    # Linear layer to find initial cell state of LSTM
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)                    # Linear layer to create a gating scalar
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)                         # Linear layer to find scores over vocabulary
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        encoder_out: (batch_size, num_pixels, encoder_dim)
        encoded_captions: (batch_size, max_caption_length)
        caption_lengths: (batch_size)
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Embeddings
        embeddings = self.embedding(encoded_captions)                 # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)                    # (batch_size, decoder_dim)

        # Set decode length excluding <END> / padding
        decode_lengths = [length - 1 for length in caption_lengths]
        max_decode_length = max(decode_lengths)

        # Create tensors to hold predicted scores and attention weights
        predictions = torch.zeros(batch_size, max_decode_length, vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max_decode_length, encoder_out.size(1)).to(encoder_out.device)

        for t in range(max_decode_length):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))          # gating scalar (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding

            # LSTM cell input: [embedding, attention_weighted_encoding]
            lstm_input = torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1)
            h, c = self.decode_step(lstm_input, (h[:batch_size_t], c[:batch_size_t]))

            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas
