import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import os

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_operation = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2, inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,  stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.conv_operation(x)

class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.doubleconv = DoubleConv(in_channels=in_channels, out_channels=out_channels)
        # maxpooling membagi ukuran image jadi 2 kali lebih kecil atau setengahnya
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.doubleconv(x)
        pool = self.pool(down)
        # print(down.shape, pool.shape, "downsampling")
        return down, pool

class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.doubleconv = DoubleConv(in_channels=in_channels, out_channels=out_channels)
        # transpose convolution
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=2, stride=2)
    
    def crop_tensor(self, tensor, target_tensor):
        target_size = target_tensor.size()[2:]
        tensor_size = tensor.size()[2:]
        delta_h = tensor_size[0] - target_size[0]
        delta_w = tensor_size[1] - target_size[1]
        tensor = tensor[:, :, delta_h // 2:tensor_size[0] - delta_h // 2, delta_w // 2:tensor_size[1] - delta_w // 2]
        return tensor
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.crop_tensor(x2, x1)
        x = torch.cat([x2, x1], dim=1)
        return self.doubleconv(x)

class UNETModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.down1 = DownSampling(in_channels, 64)
        self.down2 = DownSampling(64, 128)
        self.down3 = DownSampling(128, 256)
        self.down4 = DownSampling(256, 512)
        
        self.bottle_neck = DoubleConv(512, 1024)
        
        self.up1 = UpSampling(1024, 512)
        self.up2 = UpSampling(512, 256)
        self.up3 = UpSampling(256, 128)
        self.up4 = UpSampling(128, 64)
        
        self.finalConv = nn.Conv2d(64, num_classes, 1, 1)
    
    
    def forward(self, x):
        down1, pool1 = self.down1(x)
        down2, pool2 = self.down2(pool1)
        down3, pool3 = self.down3(pool2)
        down4, pool4 = self.down4(pool3)
        
        bn = self.bottle_neck(pool4)
        
        up1 = self.up1(bn, down4)
        up2 = self.up2(up1, down3)
        up3 = self.up3(up2, down2)
        up4 = self.up4(up3, down1)
        
        return self.finalConv(up4)


class CustomDataset(Dataset):
    def __init__(self, root_path):
        super().__init__()
        self.root_path = root_path
        self.images_path = [f"{root_path}/images/{i}" for i in os.listdir(f"{root_path}/images")]
        self.masks_path = [f"{root_path}/masks/{i}" for i in os.listdir(f"{root_path}/masks")]
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
    
    def __getitem__(self, index):
        image = Image.open(self.images_path[index]).convert('RGB')
        mask = Image.open(self.masks_path[index]).convert('L')
        
        return self.transform(image), self.transform(mask)
    
    def __len__(self):
        return len(self.images_path)

    
class MyLoader(DataLoader):
    def __init__(self, dataset):
        super().__init__(dataset)

    def load(self):
        return iter(self)
    
def trainOneEpoch(trainLoader):
    pass

def testOneEpoch(testLoader):
    pass