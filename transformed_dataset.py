import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

transformers = [
    transforms.GaussianBlur((5,5)),
    transforms.RandomRotation((-180,180)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.ColorJitter(0.8,0.8),
    transforms.RandomPerspective(),
    transforms.RandomInvert(),
    transforms.AutoAugment(),
]

class TransformedImagesDataset(Dataset):
    def __init__(self, data_set: Dataset):
        self.dataset = data_set

        self.items = []
        for index, image in enumerate(self.dataset):
            for shifted_index in range(index, index + len(transformers) + 1 + 1):
                image, *rest = self.dataset[index]
                image, transformation_name = augment_image(image, shifted_index)
                self.items.append((image, *rest, transformation_name, shifted_index))

    def __getitem__(self, index: int):
        return self.items[index]

    def __len__(self):
        return len(self.dataset) * (len(transformers) + 2)

def augment_image(img: torch.Tensor, index: int) -> tuple[torch.Tensor, str]:
    global transformers
    transformation_name = ""
    number_transformers = len(transformers)
    v = index % (number_transformers+1)
    if img.dtype == torch.float32:
        img = (img * 255).to(torch.uint8)

    match v:
        case 0:
            transformation_name = "original"
        case index if 0 < index <= number_transformers:
            transformer: torch.nn.Module = transformers[index-1]
            transformation_name = type(transformer).__name__
            img = transformer(img)
    img = img.to(torch.float32) / 255.0
    return img, transformation_name
