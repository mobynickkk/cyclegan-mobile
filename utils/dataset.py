from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ImgDataset(Dataset):
    """Dataset for CycleGAN"""

    def __init__(self, files_a, files_b):
        super().__init__()
        self.files_a = files_a
        self.files_b = files_b
        self.length = len(files_a)

    def __len__(self):
        return self.length

    @staticmethod
    def image_loader(img_a, img_b):
        img_a = Image.open(img_a)
        img_a.load()
        img_b = Image.open(img_b)
        img_b.load()
        loader = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor()
        ])
        img_a = loader(img_a)
        img_b = loader(img_b)
        return img_a, img_b

    def __getitem__(self, index):
        a, b = self.image_loader(self.files_a[index], self.files_b[index])
        return a, b
