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
        self.imagepath_points_list = []
    
    def generate_data(self):
        for i in os.listdir(self.src_path):
            if i.endswith(".json"):
                pth = f"{self.src_path}/{i}"
                break
        with open(pth, 'r') as file:
            data = json.load(file)
        
        return data
    
    def __plugAnotations(self):
        for image_data in self.images:
            lst = []
            for anot_data in self.anotations:
                if anot_data[self.anot_key] == image_data['id']:
                    lst.append(
                        {
                            'file_name': image_data['file_name'],
                            'class': anot_data['category_id'],
                            'width': image_data['width'],
                            'height': image_data['height'],
                            'segmentation': anot_data['segmentation']
                        }
                    )
            self.imagepath_points_list.append(lst)
        
        return True
    
    
    def maskingLayer(self):
        if self.__plugAnotations():
            image_mask = []
            for segment_properties in self.imagepath_points_list:
                
                class_list = []
                for i in range(6):
                    class_list.append(np.zeros(shape=(segment_properties[0]['width'], segment_properties[0]['height']), dtype=np.uint8))
                    
                for segment_item in segment_properties:
                    points = np.array(segment_item['segmentation'])[0].reshape(-1, 2)
                    points = points.astype(np.int32)
                    
                    cv.fillPoly(class_list[segment_item['class']], [points], color=(255, 255, 255))
                    
                image_mask.append(class_list)
                print(f"done creating mask in f{segment_properties[0]['file_name']}")
            image_mask = np.array(image_mask)
            
            return image_mask
            
    
    def createAnotation(self, migrate_files=False):
        if self.__plugAnotations():
            for data in self.imagepath_points_list:
                img = cv.imread(f"{self.src_path}/{data[0]['file_name']}")
                width = data[0]['width']
                height = data[0]['height']
                mask = np.zeros(shape=(width, height, 3), dtype=np.uint8)
                
                for item in data:
                    
                    points = np.array(item['segmentation'])[0].reshape(-1, 1, 2)
                    points = points.astype(np.int32)
                    
                    cv.polylines(mask, [points], isClosed=True, color=((255, 0, 0)), thickness=1)
                    
                    if item['class'] == 0 or item['class']:
                        cv.fillPoly(mask, [points], color=((255, 0, 0)))
                    if item['class'] == 2:
                        cv.fillPoly(mask, [points], color=((255, 255, 0)))
                    if item['class'] == 3:
                        cv.fillPoly(mask, [points], color=((255, 255, 255)))
                    if item['class'] == 4:
                        cv.fillPoly(mask, [points], color=((0, 255, 0)))
                    if item['class'] == 5:
                        cv.fillPoly(mask, [points], color=((255, 0, 0)))
                    
                    print(item['class'])
                    cv.polylines(img, [points], isClosed=True, color=((0, 255, 255)), thickness=2)
                
                
                mask_name = f"{self.destination_path}/masks/mask_{data[0]['file_name']}"
                view_name = f"{self.destination_path}/view/view_{data[0]['file_name']}"
                    
                plt.imsave(f"{mask_name}", mask, cmap='gray')
                plt.imsave(f"{view_name}", img, cmap='gray')
                
                
                if migrate_files:
                    shutil.move(f"{self.src_path}/{item['file_name']}", f"{self.destination_path}/images/{item['file_name']}")
                else:
                    shutil.copy(f"{self.src_path}/{item['file_name']}", f"{self.destination_path}/images/{item['file_name']}")
        else:
            print("ya ga kekonfig")
            
            

def main():
    testing = CreateAnotationLayers(src_path="test_baru", destination_path="raw_data/test")
    mask = testing.maskingLayer()
    print(mask.shape)

if __name__ == "__main__":
    main()