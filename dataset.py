import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from vocabulary import Vocabulary
import torchvision.transforms as transforms

class Flickr8kDataset(Dataset):
    def __init__(self, root_dir, caption_file, transform=None, freq_threshold=5, vocab=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(caption_file)
        # Assuming columns are 'image' and 'caption'
        if 'image' not in self.df.columns or 'caption' not in self.df.columns:
            # Try alternative column names if standard CSV format differs
            self.df.columns = ['image', 'caption']

        self.imgs = self.df['image'].tolist()
        self.captions = self.df['caption'].tolist()
        self.transform = transform

        # Initialize or use vocabulary
        if vocab is None:
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build_vocabulary(self.captions)
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img_path = os.path.join(self.root_dir, img_id)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback dummy image if file is missing (helpful for testing/dummy data)
            image = Image.new("RGB", (224, 224), color="gray")

        if self.transform is not None:
            image = self.transform(image)

        numericalized_caption = [self.vocab.stoi["<START>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<END>"])

        return image, torch.tensor(numericalized_caption)

class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = torch.nn.utils.rnn.pad_sequence(targets, padding_value=self.pad_idx, batch_first=True)
        return imgs, targets

def get_loader(root_dir, caption_file, transform, batch_size=32, shuffle=True, num_workers=2, freq_threshold=5, vocab=None):
    dataset = Flickr8kDataset(root_dir, caption_file, transform=transform, freq_threshold=freq_threshold, vocab=vocab)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=Collate(pad_idx=pad_idx)
    )
    return loader, dataset.vocab
