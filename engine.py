import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import pandas as pd
import random

import matplotlib.pyplot as plt
import time
import matplotlib
matplotlib.use('Agg')

def load_model(path):
    weights = torchvision.models.EfficientNet_V2_M_Weights.DEFAULT
    auto_transforms = weights.transforms()
    # model = torchvision.models.efficientnet_v2_m()

    # output_shape = 2
    # model.classifier = torch.nn.Sequential(
    #     torch.nn.Dropout(p=0.2, inplace=True),
    #     torch.nn.Linear(in_features=1280,
    #                     out_features=output_shape,
    #                     bias=True))
    model = torch.load(path, map_location=torch.device('cpu'))
    model.eval()
    return {'model': model, 'transforms': auto_transforms}

def classify_image(classifier_dict, image_path):
    """Классифицировать изображение с помощью обученной модели."""
    # Предобработка изображения с помощью torchvision.transforms
    my_image_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        # transforms.Resize(size=(64, 64)),
        # transforms.ToTensor(),
    ])
    with Image.open(image_path) as f:
    # image = StandardScaler().fit_transform(image)
        tensor_image = classifier_dict['transforms'](my_image_transform(f))
    tensor_image = tensor_image.unsqueeze(0)  # Добавляем размерность batch

    start_time = time.time()

    with torch.no_grad():
        output = classifier_dict['model'](tensor_image)
        score = round(torch.softmax(output, dim=1).tolist()[0][1], 2)
        print(score)
        predicted_class = output.argmax(dim=1)

    time_taken = round(time.time() - start_time, 4)
    return {'score': score, 'time_taken': time_taken}

def blood_pressure_estimation(sid):
    df = pd.read_csv('data/external/targets.csv')
    syst_estimated = df.query(f'sid == {int(sid)}').syst_estimated.tolist()[0]
    diast_estimated = df.query(f'sid == {int(sid)}').diast_estimated.tolist()[0]
    syst_target = df.query(f'sid == {int(sid)}').systbp.tolist()[0]
    diast_target = df.query(f'sid == {int(sid)}').diastbp.tolist()[0]

    if syst_estimated < 120 and diast_estimated < 80:
        condition = 'Normal'
    elif syst_estimated >= 180 or diast_estimated >= 120:
        condition = 'Hypertensive Crisis'
    elif syst_estimated >= 120 and syst_estimated <= 129 \
        and diast_estimated < 80:
        condition = 'Elevatied'
    elif syst_estimated >= 130 and syst_estimated <= 139 \
        or diast_estimated >= 80 and diast_estimated <= 89:
        condition = 'Hypertension Stage 1'
    elif syst_estimated >= 140 or diast_estimated >= 90:
        condition = 'Hypertension Stage 2'
    else:
        condition = 'Wrong estimation, condition is unknown'

    return {'syst_estimated': round(syst_estimated, 2),
            'diast_estimated': round(diast_estimated, 2),
            'syst_target': syst_target,
            'diast_target': diast_target, 
            'condition': condition}