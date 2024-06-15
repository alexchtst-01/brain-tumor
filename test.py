import os 

root = "./raw_data/test"

for i in os.listdir(root):
    pth = i.endswith('.json')
    print(pth)