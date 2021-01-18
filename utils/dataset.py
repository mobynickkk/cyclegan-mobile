import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path


class ImgDataset(Dataset):
    """Dataset for CycleGAN"""

    def __init__(self, files_a_dir, files_b_dir):
        super().__init__()
        self.files_a = list(Path(files_a_dir).rglob('*.jpg'))
        self.files_b = list(Path(files_b_dir).rglob('*.jpg'))
        self.length = min(len(self.files_a), len(self.files_b))

    def __len__(self):
        return self.length

    @staticmethod
    def image_loader(img_a, img_b):
        if not torch.cuda.is_available():
            raise BaseException('GPU is not available')
        device = torch.device('cuda')
        img_a = Image.open(img_a)
        img_a.load()
        img_b = Image.open(img_b)
        img_b.load()
        loader = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor()
        ])
        img_a = loader(img_a).to(device)
        img_b = loader(img_b).to(device)
        return img_a, img_b

    def __getitem__(self, index):
        a, b = self.image_loader(self.files_a[index], self.files_b[index])
        return a, b
