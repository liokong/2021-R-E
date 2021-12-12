import scipy.stats as stats
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2
import torch
import torch.nn as nn

import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
import time

bmd=np.load("bmd820.npy")
bmd = np.delete(bmd, [208,235,322,597,675,757,758,767,773,774], axis=0)
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1

CT = np.load('spine.npy')

ref_bmd = 0.9747
std = 0.1185
tsc = (bmd - ref_bmd)/std
label = (tsc<-2.5).astype(np.uint8)


num = np.random.randint(1,20)
CT_train, CT_valid, y_train, y_valid = train_test_split(CT, label, test_size=0.4, random_state=num)
CT_valid, CT_test, y_valid, y_test = train_test_split(CT_valid, y_valid, test_size=0.5, random_state=num)
print(CT_train.shape, CT_valid.shape, CT_test.shape)
'''
def aug(X_train):
  b=[]
  a=np.zeros((X_train.shape[0],256,256,1))
  for i in range(X_train.shape[0]):
    a[i,:,:,0]=cv2.flip(X_train[i,:,:,0],1)
  b.append(a)

  a=np.zeros((X_train.shape[0],256,256,1))
  for i in range(X_train.shape[0]):
    a[i,:,:,0]=cv2.flip(X_train[i,:,:,0],0)
  b.append(a)

  for j in range(1,12):
    a=np.zeros((X_train.shape[0],256,256,1))
    for i in range(X_train.shape[0]):
      a[i,:,:,0]=cv2.warpAffine(X_train[i,:,:,0], cv2.getRotationMatrix2D((128,128), j*30, 1), (256, 256))
    b.append(a)

  for j in [-4,-3,-2,-1,1,2,3,4]:
    a=np.zeros((X_train.shape[0],256,256,1))
    for i in range(X_train.shape[0]):
      a[i,:,:,0]=cv2.warpAffine(X_train[i,:,:,0], np.float32([[1,0,j*20],[0,1,0]]), (256, 256))
    b.append(a)
    
  for i in b:
    X_train=np.concatenate((X_train,i),axis=0)
  
  return X_train

CT_train = aug(CT_train)

y_train = np.repeat([y_train],22, axis=0).reshape(-1)

'''
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, ct, bmd, transform=None):
        self.ct = ct
        self.bmd = bmd
        self.transform = transform
    def __len__(self):
        return len(self.ct)
    
    def __getitem__(self, index):
        img = self.ct[index]
        
        if self.transform:
            img = self.transform(img).view([1,256,256])
        return img, np.array([self.bmd[index]])

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_ct_dataset = CustomDataset(CT_train, y_train, transform=train_transform)
valid_ct_dataset = CustomDataset(CT_valid, y_valid, transform=train_transform)
test_ct_dataset = CustomDataset(CT_test, y_test, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_ct_dataset, batch_size=16, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_ct_dataset, batch_size=2, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ct_dataset, batch_size=2, shuffle=True)

device='cuda' if torch.cuda.is_available() else 'cpu'
'''
start_layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1,bias=False)
model = nn.Sequential(start_layer, torchvision.models.resnet18(pretrained=True))

num_ftrs = model[1].fc.in_features
model[1].fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 256),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.Dropout(0.1),
    nn.Linear(128, 1),
    nn.Sigmoid()
)
model.cuda()


def fit(model, criterion, optimizer, epochs, train_loader, valid_loader):
    model.train()
    train_loss = 0
    train_acc = 0
    train_correct = 0
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    for epoch in range(epochs):
        start = time.time()

        for train_x, train_y in train_loader:
            model.train()
            train_x, train_y = train_x.to(device).float(), train_y.to(device).float()
            optimizer.zero_grad()
            pred = model(train_x)
            loss = criterion(pred, train_y)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            y_pred = pred.cpu()
            y_pred[y_pred >= 0.5] = 1
            y_pred[y_pred < 0.5] = 0
            train_correct += y_pred.eq(train_y.cpu()).int().sum()
            
        # validation data check
        valid_loss = 0
        valid_acc = 0
        valid_correct = 0

        for valid_x, valid_y in valid_loader:
            with torch.no_grad():
                model.eval()
                valid_x, valid_y = valid_x.to(device).float(), valid_y.to(device).float()
                pred = model(valid_x)
                loss = criterion(pred, valid_y)
            valid_loss += loss.item()
            y_pred = pred.cpu()
            y_pred[y_pred >= 0.5] = 1
            y_pred[y_pred < 0.5] = 0
            valid_correct += y_pred.eq(valid_y.cpu()).int().sum()

        train_acc = train_correct/len(train_loader.dataset)
        valid_acc = valid_correct/len(valid_loader.dataset)
        
        print(f'{time.time() - start:.3f}sec : [Epoch {epoch+1}/{epochs}] -> train loss: {train_loss/len(train_loader):.4f}, train acc: {train_acc*100:.3f}% / valid loss: {valid_loss/len(valid_loader):.4f}, valid acc: {valid_acc*100:.3f}%')

        train_losses.append(train_loss/len(train_loader))
        train_accuracies.append(train_acc)
        valid_losses.append(valid_loss/len(valid_loader))
        valid_accuracies.append(valid_acc)
        train_loss = 0
        train_acc = 0
        train_correct = 0

        torch.save(model, f'model/predict{epoch+1}.pt')

    acc=train_accuracies
    val_acc=valid_accuracies
    loss=train_losses
    val_loss=valid_losses  

    epo = range(1, len(acc)+1)
    plt.plot(epo, loss, 'b', label="Traing loss")
    plt.plot(epo, val_loss, 'r', label="Val loss")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('predictSick_loss.png')
    plt.show()  
    
    plt.plot(epo, acc, 'b', label="Traing accuracy")
    plt.plot(epo, val_acc, 'r', label="Val accuracy")
    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig('predictSick__acc.png')
    plt.show()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
fit(model, criterion, optimizer, 30, train_loader, valid_loader)
'''
for i in range(50):
    model = torch.load(f'model/predict{i+1}.pt')
    criterion = nn.MSELoss()
    def eval(model, criterion, test_loader):
        with torch.no_grad():
            model.eval()
            correct = 0
            losses = 0
            for test_x, test_y in test_loader:
                test_x, test_y = test_x.to(device).float(), test_y.to(device).float()
                pred = model(test_x)
                loss = criterion(pred, test_y)
                y_pred = pred.cpu()
                y_pred[y_pred >= 0.5] = 1
                y_pred[y_pred < 0.5] = 0
                losses += loss.item()
                correct += y_pred.eq(test_y.cpu()).int().sum()
        print(f'eval loss: {losses/len(test_loader):.4f}, eval acc: {correct/len(test_loader.dataset)*100:.3f}%')   

    eval(model, criterion, test_loader)
