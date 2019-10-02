import argparse
import numpy as np
import torch
from torch import nn,optim
import torch.nn.functional as F
from torchvision import datasets,transforms,models
import json
from collections import OrderedDict
from PIL import Image

from utility_func import process_image
from func import predict, load_checkpoint

parser=argparse.ArgumentParser(description='Predict using trained model')

parser.add_argument('image_path',action='store',
                    help='Please provide image path to be processed, default image is already given')

parser.add_argument('-m', '--model', action='store', dest='model_arch', default='vgg16',
                    help='Please provide model architecture, Recommended and default:"vgg16"')

parser.add_argument('-s','--save_dir', action='store', dest='save_directory', default='my_chkpt.pth',
                    help='Please provide path where model is saved')

parser.add_argument('-g','--gpu', action='store_true', default=False, dest='mode',
                    help='Please give mode in which model will be running GPU or CPU, default is CPU')

parser.add_argument('-t','--top_k', action='store', dest='topk', type=int, default = 5,
                    help='Please provide number of most likely classes to be displayed')

parser.add_argument('-c','--cat_to_name', action='store', dest='cat_name_path', default = 'cat_to_name.json',
                    help='Please provide path of category to names file')

args=parser.parse_args()

image_path=args.image_path
save_dir=args.save_directory
mode=args.mode
topk=args.topk

cat_to_name=args.cat_name_path
with open(cat_to_name, 'r') as f:
    cat_to_name = json.load(f)

load_pretrained_model=args.model_arch
model=getattr(models,load_pretrained_model)(pretrained=True)

loaded_model=load_checkpoint(model,save_dir,mode)

processed_img=process_image(image_path)

probs_list, classes_list=predict(processed_img,loaded_model,topk)

print(probs_list)
print(classes_list)

names = []
for i in classes_list:
    names.append(cat_to_name[i])
    
print('Most likely flower is {} with proabablity percentage: {}'.format(names[0],(round(probs_list[0]*100,4))))

