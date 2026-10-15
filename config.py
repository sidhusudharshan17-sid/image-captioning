import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data paths (pointing to Flickr8k dataset if downloaded, or customizable)
DATA_DIR = './data/flickr8k'
IMAGE_DIR = 'c:/omniroute/imagecaptioning/Flickr_8K_1000'
CAPTION_FILE = 'c:/omniroute/imagecaptioning/Flickr_8K_1000/captions.txt'

# Model hyperparameters
EMBED_SIZE = 256          # Dimension of word embeddings
ATTENTION_DIM = 256       # Dimension of attention layer
DECODER_DIM = 512         # Dimension of LSTM hidden state
ENCODER_DIM = 2048        # ResNet101 feature dimension (spatial feature map size)
VOCAB_SIZE = 5000         # Maximum vocabulary size
MAX_LEN = 40              # Maximum length of generated caption

# Training hyperparameters
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 1e-4
PRINT_FREQ = 100
SAVE_DIR = './checkpoints'
