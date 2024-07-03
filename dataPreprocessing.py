import json
import os 
import shutil
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2 as cv

class CreateAnotationLayers:
    def __init__(self, src_path, destination_path, anot_key='image_id'):
        self.__showWarning()
        self.src_path = src_path
        self.destination_path = destination_path
        self.data = self.generate_data()
        self.images = self.data['images']
        self.anotations = self.data['annotations']
        self.anot_key = anot_key
        self.__image_segmentations = []
    
    def generate_data(self):
        for i in os.listdir(self.src_path):
            if i.endswith(".json"):
                pth = f"{self.src_path}/{i}"
                break
        with open(pth, 'r') as file:
            data = json.load(file)
        
        return data
    
    def __showWarning(self):
        print("make sure your surce and destination like below: ")
        print("source ")
        print("  --there is coco.json ")
        print("  --and the images ")
        print("destination ")
        print("  --images ")
        print("  --masks ")
        print("  --view ")
    
    def __plugAnotations(self):
        for image in self.images:
            lst = []
            for anotation in self.anotations:
                if image['id'] == anotation[self.anot_key]:
                    lst.append({
                        'file_name': image['file_name'],
                        'class': anotation['category_id'],
                        'width': image['width'],
                        'height': image['height'],
                        'points': anotation['segmentation']
                    })
            self.__image_segmentations.append(lst)
        
        return True
    
    def createAnotation(self, migrate_all_files=False):
        if self.__plugAnotations():
            for segment_properties in tqdm(self.__image_segmentations):
                img = cv.imread(f"{self.src_path}/{segment_properties[0]['file_name']}")
                width = segment_properties[0]['width']
                height = segment_properties[0]['width']
                
                mask = np.zeros(shape=(width, height, 3), dtype=np.uint8)
                
                for segment_item in segment_properties:
                    points = np.array(segment_item['points'])[0].reshape(-1, 1, 2)
                    points = points.astype(np.int32)
                    
                    cv.polylines(mask, [points], isClosed=True, color=((255, 0, 0)), thickness=1)
                    
                    if segment_item['class'] == 0 or segment_item['class']:
                        cv.fillPoly(mask, [points], color=((255, 0, 0)))
                    if segment_item['class'] == 2:
                        cv.fillPoly(mask, [points], color=((255, 255, 0)))
                    if segment_item['class'] == 3:
                        cv.fillPoly(mask, [points], color=((255, 255, 255)))
                    if segment_item['class'] == 4:
                        cv.fillPoly(mask, [points], color=((0, 255, 0)))
                    if segment_item['class'] == 5:
                        cv.fillPoly(mask, [points], color=((255, 0, 0)))
                    
                    cv.polylines(img, [points], isClosed=True, color=((0, 255, 255)), thickness=2)
                    
                mask_name = f"{self.destination_path}/masks/mask_{segment_properties[0]['file_name']}"
                view_name = f"{self.destination_path}/view/view_{segment_properties[0]['file_name']}"
                    
                plt.imsave(f"{mask_name}", mask, cmap='gray')
                plt.imsave(f"{view_name}", img, cmap='gray')
                
                if migrate_all_files:
                    shutil.move(f"{self.src_path}/{segment_item['file_name']}", f"{self.destination_path}/images/{segment_item['file_name']}")
                else:
                    shutil.copy(f"{self.src_path}/{segment_item['file_name']}", f"{self.destination_path}/images/{segment_item['file_name']}")
        else:
            print("ya kodenya error, kasian")

