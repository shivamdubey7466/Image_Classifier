import argparse
import numpy as np
import torch
from torch import nn,optim
import torch.nn.functional as F
from torchvision import datasets,transforms,models

from utility_func import load_data
from func import classifier, train_network, test_network, save, load_checkpoint

parser=argparse.ArgumentParser(description='Network Training')

parser.add_argument('data_directory', action='store',
                    help='Please provide data directory path for training')

parser.add_argument('-s','--save_dir', action='store', dest='save_directory', default='my_chkpt.pth',
                    help='Please provide path of saving directory,default is "my_chkpt.pth" ')

parser.add_argument('-m', '--model', action='store', dest='model_arch', default='vgg16',
                    help='Please provide model architecture, Recommended and default:"vgg16"')

parser.add_argument('-e', '--epochs', action='store',dest='epochs', type=int, default=10,
                    help='Please provide number of epochs,default is 10')

parser.add_argument('-lr','--learning_rate', action='store', dest='learning_rate', type=float, default=0.001,
                    help='Please provide learning rate,default is 0.001')

parser.add_argument('-g','--gpu', action='store_true', default=False, dest='mode',
                    help='Please give mode in which model will be running GPU or CPU, default is CPU')

parser.add_argument('-d', '--dropout', action='store', type=float, default=0.05, dest='dropout',
                    help='Please provide dropout value,default is 0.05')

parser.add_argument('-hi','--hidden_inputs', action='store', type=int, default=1024, dest='hidden_inputs',
                    help='Please provide number of hidden nodes')


args=parser.parse_args()

data_dir=args.data_directory
save_dir=args.save_directory
epochs=args.epochs
lr=args.learning_rate
drop_prob=args.dropout
hidden_nodes=args.hidden_inputs
mode=args.mode

trainloader, testloader, validloader, train_datasets, test_datasets, valid_datasets=load_data(data_dir)

load_pretrained_model=args.model_arch
model=getattr(models,load_pretrained_model)(pretrained=True)

input_nodes=model.classifier[0].in_features
classifier(model,input_nodes,hidden_nodes,drop_prob)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr)

model, optimizer = train_network(model,epochs,mode,trainloader,validloader,criterion,optimizer)

test_network(model,mode,testloader,criterion)

save(model,optimizer,epochs,save_dir,train_datasets)

print("Ready for prediction...")

