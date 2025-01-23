import torch
import os
from torch.utils.data import DataLoader
from dataset import dataset
from para import net,mod,epo,batch,l1
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = 'save/{}/'.format(mod)
net.to(device)
#criterion = nn.MSELoss().to(device)
criterion = nn.SmoothL1Loss(beta=1.0).to(device)
#criterion = torch.nn.L1Loss().to(device)
#criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3,betas=(0.95, 0.999))
scheduler = MultiStepLR(optimizer, milestones=[0.25*epo,0.5*epo,0.75*epo], gamma=0.2)  # learning rates
train_dataset = dataset("Data", test=False)
trainloader = DataLoader(train_dataset,
                        batch_size = batch,
                        shuffle = True,
                        num_workers = 0,
                        drop_last=True)

running_loss = 0
running_loss1 = 0
os.makedirs(save_path, exist_ok=True)
save_path = 'save/{}/'.format(mod)

for epoch in  tqdm(range(epo)): 
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        lonlat, inpu, label, mask = data
        label = label.to(device)
        mask = mask.to(device)
        inpu = inpu.to(device)
        optimizer.zero_grad()
        output =  net(inpu)
        loss1 = criterion(output*mask, label)

        l1_regularization = 0
        for param in net.parameters():
            l1_regularization += torch.sum(torch.abs(param))
        lambda_value = l1 
        loss = lambda_value * l1_regularization + loss1

        running_loss = running_loss+loss
        running_loss1 = running_loss1+loss1
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_loss1 += loss1.item()
        if (i+1) % 10 == 0:
            print('train all 300 [%d,%5d] loss: %0.3f  loss1: %0.3f ' % (epoch + 1, i + 1, loss.item(), loss1.item()))

    if (epoch+1) % 10 == 0:
        torch.save(net.state_dict(), save_path+'model_{}.pth'.format(epoch+1))
        print("Saved model weights.")
    if (epo - (epoch+1))<25:
        torch.save(net.state_dict(), save_path+'model_{}.pth'.format(epoch+1))
        print("Saved model weights.")
    running_loss=0
    running_loss1=0


        