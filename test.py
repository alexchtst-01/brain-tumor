from TrainingProperties import UNETModel, CustomDataset, MyLoader
import torch

x = torch.rand(1, 3, 512, 512)
model = UNETModel()
model(x)
