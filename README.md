# Segmentation Project by Alex Cinatra Hutasoit

## Description
This program include 4 files that has it's own task : 
|  |  |
|--|--|
| `TrainingProperties.py` | to define the deep learning model and create the other requires properties such as function and class |
| `dataPreprocessing.py` | to clean the data |
| `interface.py` | to present the result of the model prediction |

## Program Plan Development
| development process | status |
|--|--|
| create the model | 40% |
| data cleaning | 100% |

## How to use
**`dataPreprocessing.py`**
make sure that your `dest_data_path` and `src_data_path` is has the following structure:

`dest_data_path`
src_data_path
│
├── train
│   ├── images
│   │   ├── images.jpg
│   │   └── .....
│   └── masks
│   │   ├── masks.jpg
│   │   └── .....
│
├── test
│   ├── images
│   │   ├── images.jpg
│   │   └── .....
│   └── masks
│   │   ├── masks.jpg
│   │   └── .....
│
└── manual_test `[opsional]`

`src_data_path`
src_data_path
│
├── train
│   ├── anotation.json
│   ├── images.jpg
│   └── ....
│
├── test
│   ├── anotation.json
│   ├── images.jpg
│   └── ....
│
└── manual_test `[opsional]`
│   ├── anotation.json
│   ├── images.jpg
│   └── ....