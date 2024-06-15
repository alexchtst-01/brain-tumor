import json
import os 
import shutil
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2 as cv

class CreateAnotationLayers:
    def __init__(self, src_path, destination_path, anot_key='image_id'):
        self.src_path = src_path
        self.destination_path = destination_path
        self.data = self.generate_data()
        self.images = self.data['images']
        self.anotations = self.data['annotations']
        self.anot_key = anot_key
        self.__imagepath_points_list = []
    
    def generate_data(self):
        for i in os.listdir(self.src_path):
            if i.endswith(".json"):
                pth = f"{self.src_path}/{i}"
                break
        with open(pth, 'r') as file:
            data = json.load(file)
        
        return data
    
    def __checklen__(self):
        return len(self.data['images']) == len(self.data['annotations'])
    
    def __checkCorrespondItem__(self):
        n = 0
        for image_data, anot_data in zip(self.images, self.anotations):
            if image_data['id'] == anot_data[self.anot_key]:
                self.__imagepath_points_list.append(
                    {
                        'file_name': image_data['file_name'],
                        'width': image_data['width'],
                        'height': image_data['height'],
                        'segmentation': anot_data['segmentation']
                    }
                )
                n += 1
        return n == len(self.data['images'])
    
    
    def createAnotation(self, migrate_all_files=False):
        if self.__checkCorrespondItem__() and self.__checklen__():
            for item in tqdm(self.__imagepath_points_list):
                width = item['width']
                height = item['height']
                points = np.array(item['segmentation'])[0].reshape(-1, 2)
                points = points.astype(np.int32)
                
                mask = np.zeros(shape=(width, height))
                img = cv.imread(f"{self.src_path}/{item['file_name']}")
                
                cv.polylines(mask, [points], isClosed=True, color=255, thickness=1)
                cv.fillPoly(mask, [points], color=255)
                cv.polylines(img, [points], isClosed=True, color=255, thickness=2)
                
                mask_name = f"{self.destination_path}/masks/mask_{item['file_name']}"
                view_name = f"{self.destination_path}/view/view_{item['file_name']}"
                plt.imsave(f"{mask_name}", mask, cmap='gray')
                plt.imsave(f"{view_name}", img, cmap='gray')
                
                if migrate_all_files:
                    shutil.move(f"{self.src_path}/{item['file_name']}", f"{self.destination_path}/images/{item['file_name']}")
                else:
                    shutil.copy(f"{self.src_path}/{item['file_name']}", f"{self.destination_path}/images/{item['file_name']}")
        else:
            print("configurasi data tidak sesuai")

# testingMigration = CreateAnotationLayers(src_path="raw_data/test", destination_path="data/test")
# testingMigration.createAnotation()

# trainingMigration = CreateAnotationLayers(src_path="raw_data/train", destination_path="data/train", anot_key='id')
# trainingMigration.createAnotation()