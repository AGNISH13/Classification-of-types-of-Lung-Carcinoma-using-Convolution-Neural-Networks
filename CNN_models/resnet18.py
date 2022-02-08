

import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transf
import numpy as np
import torch.nn as  nn
import copy
import matplotlib.pyplot as plt



def res_net(folder_path):
  
  
  # Setting device

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Performing Data Augmentation for increased accuracy

  train_transform = transf.Compose([
      transf.Resize((224,224)),
      transf.RandomHorizontalFlip(p=0.5),
      transf.RandomRotation(degrees=45),
      transf.ColorJitter(brightness=0.5),
      transf.RandomVerticalFlip(p=0.05),
      transf.ToTensor()
  ])

  val_transform = transf.Compose([
      transf.Resize((224,224)),
      transf.ToTensor()
  ])

  # Hyperparameter tuning

  epoch_no=20
  l_rate = 0.00001
  batch_size = 30

  # Loading the dataset in the system

  train_ds = ImageFolder(folder_path+'/test',transform=train_transform)
  val_ds = ImageFolder(folder_path+'/val', transform = val_transform)
  train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
  val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

  # Training and Validating with fine tuned ResNet model 

  def train_model(model, criterion, optim, epoch_no):
    train_loss_values = []
    val_loss_values = []
    best_acc= 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(epoch_no):
      running_loss = 0.0
      running_acc = 0.0
      model.train()
      for images,labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(True):
          outputs = model(images)
          _ ,preds = torch.max(outputs,1)
          loss = criterion(outputs,labels)
          loss.backward()
          optim.step()
        optim.zero_grad()

      # Calculating and printing all Statistics
        running_loss += loss.item()*batch_size
        running_acc += torch.sum(preds==labels)
      running_val_loss, running_val_acc = model_val(model, criterion, optim)
      epoch_train_loss = running_loss/len(train_ds)
      epoch_train_acc = running_acc.double()/len(train_ds)
      print("Epoch: {}".format(epoch+1))
      print('-'*10)
      print('Train Loss: {:.4f}   Train Acc: {:.4f}'.format(epoch_train_loss,epoch_train_acc))
      epoch_val_loss = running_val_loss/len(val_ds)
      epoch_val_acc = running_val_acc.double()/len(val_ds)
      print('Val Loss: {:.4f}   Val Acc: {:.4f}'.format(epoch_val_loss,epoch_val_acc))
      print()
      train_loss_values.append(epoch_train_loss)
      val_loss_values.append(epoch_val_loss)
      if epoch_val_acc > best_acc:
        best_acc = epoch_val_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    # Printing Best Validation Accuracy and Loss vs Epoch Plot
    print("Best model has validation accuracy: {}".format(best_acc))
    model.load_state_dict(best_model_wts)
    plt.plot(range(epoch_no),np.array(train_loss_values),'b',label='Train Curve')
    plt.plot(range(epoch_no),np.array(val_loss_values),'g',label='Validation Curve')
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    plt.legend()

    return model


  def model_val(model, criterion):
    model.eval()
    running_val_loss = 0.0
    running_val_acc = 0.0
    for images,labels in val_loader:
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      _ ,preds = torch.max(outputs,1)
      loss = criterion(outputs,labels)
      running_val_loss += loss.item()*batch_size
      running_val_acc += torch.sum(preds==labels)
    return running_val_loss, running_val_acc

  # Initialising model and performing Loss calculation and Optimization

  model = torchvision.models.resnet18(pretrained=True)
  model.fc = nn.Linear(model.fc.in_features,3)
  model = model.to(device)
  criterion = nn.CrossEntropyLoss()
  criterion.to(device)
  optim = torch.optim.Adam(model.parameters(), lr = l_rate)

  # Training and evaluation

  print("<---ResNet CNN model---> \n\n\n")
  model = train_model(model, criterion, optim, epoch_no)