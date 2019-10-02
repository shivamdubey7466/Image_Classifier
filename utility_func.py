import numpy as np
import torch
from torch import nn,optim
import torch.nn.functional as F
from torchvision import datasets,transforms,models
from PIL import Image

def load_data(data_dir):
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms=transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    test_transforms=transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_datasets=datasets.ImageFolder(data_dir + '/train',transform=train_transforms)
    valid_datasets=datasets.ImageFolder(data_dir + '/valid',transform=valid_transforms)
    test_datasets=datasets.ImageFolder(data_dir + '/test',transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader= torch.utils.data.DataLoader(train_datasets,batch_size=50,shuffle=True)
    validloader=torch.utils.data.DataLoader(valid_datasets,batch_size=50,shuffle=True)
    testloader=torch.utils.data.DataLoader(test_datasets,batch_size=50,shuffle=True)
    
    return trainloader, testloader, validloader, train_datasets, test_datasets, valid_datasets


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image_path)
    image_transform=transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    
    img=image_transform(image)
    img=np.array(img)
    return img
