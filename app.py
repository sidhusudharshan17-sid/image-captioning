import os
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import torch
import torchvision.transforms as transforms
from PIL import Image

import config
from models import EncoderCNN, DecoderWithAttention
from caption import generate_caption
from utils import visualize_attention
import vocabulary

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load Model on Startup
device = config.DEVICE
print(f"Loading model on {device}...")

encoder = None
decoder = None
vocab = None

model_path = os.path.join(config.SAVE_DIR, "best_model.pth.tar")
if os.path.exists(model_path):
    try:
        torch.serialization.add_safe_globals([vocabulary.Vocabulary])
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        vocab = checkpoint['vocab']
        vocab_size = len(vocab)

        encoder = EncoderCNN().to(device)
        decoder = DecoderWithAttention(
            attention_dim=config.ATTENTION_DIM,
            embed_dim=config.EMBED_SIZE,
            decoder_dim=config.DECODER_DIM,
            vocab_size=vocab_size
        ).to(device)

        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['state_dict'])
        encoder.eval()
        decoder.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: Checkpoint not found at {model_path}. Please train the model first.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if encoder is None or decoder is None or vocab is None:
        return jsonify({'success': False, 'error': 'Model not loaded. Please train the model first.'}), 500

    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected image'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        try:
            # Generate caption and attention weights
            caption, attention_weights = generate_caption(encoder, decoder, vocab, image_path, device)
            caption_text = " ".join(caption)

            # Generate and save attention heatmap visualization
            output_filename = f"attention_{filename}"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            visualize_attention(image_path, caption, attention_weights, vocab, filename=output_path)

            return jsonify({
                'success': True,
                'caption': caption_text,
                'image_url': url_for('static', filename=f'uploads/{filename}'),
                'attention_url': url_for('static', filename=f'outputs/{output_filename}')
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    return jsonify({'success': False, 'error': 'Invalid file type. Allowed types: png, jpg, jpeg'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
