import numpy as np
import torch
from torch import nn,optim
import torch.nn.functional as F
from torchvision import datasets,transforms,models
from collections import OrderedDict
from PIL import Image

def classifier(model,input_nodes,hidden_nodes,drop_prob):
    
    for params in model.parameters():
        params.requires_grad=False
    
    classifier = nn.Sequential(OrderedDict([
                           ('fc1',nn.Linear(input_nodes,hidden_nodes)),
                           ('ReLu1',nn.ReLU()),
                           ('Dropout1',nn.Dropout(p=drop_prob)),
                           ('fc2',nn.Linear(hidden_nodes,102)),
                           ('output',nn.LogSoftmax(dim=1))
                           ]))
    
    model.classifier = classifier
    
    return model

def train_network(model,epochs,mode,trainloader,validloader,criterion,optimizer):
    
    if mode==True:
        model.to('cuda')
    else:
        pass
    print("Please hold on, training started")
    for e in range(epochs):
        running_loss=0
        for ii,(images,labels) in enumerate(trainloader):
            
            if mode==True:
                images,labels=images.to('cuda'),labels.to('cuda')
            else:
                pass
            optimizer.zero_grad()
            log_ps=model.forward(images)
            loss=criterion(log_ps,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()

        else:
            model.eval()
            valid_loss=0
            accuracy=0
            with torch.no_grad():
                for ii,(images,labels) in enumerate(validloader):
                    
                    if mode==True:
                        images,labels=images.to('cuda'),labels.to('cuda')
                    else:
                        pass
                    log_ps=model.forward(images)
                    valid_loss+=criterion(log_ps,labels).item()
                    ps=torch.exp(log_ps)
                    equals=(ps.max(dim=1)[1]==labels.data)
                    accuracy+=torch.mean(equals.type(torch.FloatTensor))

            model.train()

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                  "Valid Accuracy: {:.3f}".format(accuracy/len(validloader)))

    return model,optimizer

def test_network(model,mode,testloader,criterion):
    
    if mode==True:
        model.to('cuda')
    else:
        pass
    model.eval()
    with torch.no_grad():
        accuracy=0
        loss=0
        for ii,(images,labels) in enumerate(testloader):
            
            if mode==True:
                images,labels=images.to('cuda'),labels.to('cuda')
            else:
                pass
            log_ps=model.forward(images)
            loss+=criterion(log_ps,labels)
            ps=torch.exp(log_ps)
            equals=(ps.max(dim=1)[1]==labels.data)
            accuracy+=torch.mean(equals.type(torch.FloatTensor))

    print("Test Loss: {:.3f}.. ".format(loss/len(testloader)),
          "Test Accuracy %: {:.3f}".format((accuracy/len(testloader))*100))
    
def save(model,optimizer,epochs,save_dir,train_datasets):
    
    checkpoint={'classifier':model.classifier,
            'class_to_idx':train_datasets.class_to_idx,
            'model_load_state_dict':model.state_dict,
            'no_of_epochs':epochs,
            'optimizer_state_dict':optimizer.state_dict}
    
    return torch.save(checkpoint,save_dir)

def load_checkpoint(model,save_dir,mode):
    
    if mode == True:
        checkpoint = torch.load(save_dir)
    else:
        checkpoint = torch.load(save_dir, map_location=lambda storage, loc: storage)
        
    model.load_state_dict=checkpoint['model_load_state_dict']
    model.class_to_idx=checkpoint['class_to_idx']
    model.classifier=checkpoint['classifier']
    
    return model

def predict(processed_img,loaded_model,topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file   
    loaded_model.cpu()
    img = processed_img
    img = torch.from_numpy(img).type(torch.FloatTensor)
    img = img.unsqueeze_(0)

    loaded_model.eval()
    with torch.no_grad():
        log_ps = loaded_model.forward(img)

    probs = torch.exp(log_ps)
    probs_top = probs.topk(topk)[0]
    index_top = probs.topk(topk)[1]
    probs_list = np.array(probs_top)[0]
    index_list = np.array(index_top[0])
    
    class_to_idx = loaded_model.class_to_idx
    indx_to_class = {x: y for y, x in class_to_idx.items()}

    classes_list = []
    for index in index_list:
        classes_list += [indx_to_class[index]]
        
    return probs_list, classes_list