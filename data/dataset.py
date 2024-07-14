from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms

class AlignedDataset(Dataset):
    def __init__(self, root_dir, img_size=256):
        assert os.path.isdir(root_dir), f'{root_dir} is not a valid directory' 

        self.img_size = img_size
        self.root_dir = root_dir
        self.img_paths = []
        for img_file in os.listdir(root_dir):
            self.img_paths.append(os.path.join(root_dir, img_file))
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        w, h = image.size
        w2 = int(w / 2)
        input = image.crop((0, 0, w2, h))
        output = image.crop((w2, 0, w, h))

        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        input = transform(input)
        output = transform(output)

        return input, output

def test():
    dataset = AlignedDataset('dataset/original')
    input, output = dataset[0]
    print(input.shape, output.shape)

if __name__ == "__main__":
    test()
