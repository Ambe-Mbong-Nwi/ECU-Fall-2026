import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.RandomCrop((256,256),pad_if_needed=True),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.OxfordIIITPet("pets_data/","trainval",transform=transform,download=True)
train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [int(len(train_dataset)*0.8), int(len(train_dataset)*0.2)])
test_dataset = torchvision.datasets.OxfordIIITPet("pets_data/","test",transform=transform,download=True)

train_loader = DataLoader(train_dataset,batch_size=8,shuffle=True)
val_loader = DataLoader(validation_dataset,batch_size=8)
test_loader = DataLoader(test_dataset,batch_size=8)



class LitNetwork(pl.LightningModule):
    def __init__(self):
        super(LitNetwork, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.ReLU = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(128*64*64, 37)
        
        n_classes = 37
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.val_acc = torchmetrics.Accuracy("multiclass",num_classes=n_classes,average='micro')
        self.test_acc = torchmetrics.Accuracy("multiclass",num_classes=n_classes,average='micro')

    def forward(self, x):
        x = self.ReLU(self.conv1(x))
        x = self.pool1(x)
        x = self.ReLU(self.conv2(x))
        x = self.pool2(x)
        x = self.fc1(torch.flatten(x,start_dim=1))
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, data, batch_idx):
        im, label = data[0], data[1]
        outs = self.forward(im)
        loss = self.loss_func(outs, label)
        self.log("train_loss",loss,batch_size=1,sync_dist=True)
        return loss
    
    def validation_step(self, val_data, batch_idx):
        im, label = val_data[0], val_data[1]
        outs = self.forward(im)
        self.val_acc(outs,label)
        self.log("val_acc",self.val_acc,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        return None

    def test_step(self, test_data, batch_idx):
        im, label = test_data[0], test_data[1]
        outs = self.forward(test_data.im)
        self.test_acc(outs,test_data.label)
        self.log("test_acc",self.test_acc,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        return None


model = LitNetwork()
checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_acc', save_top_k=1, mode='max')
#logger = pl_loggers.TensorBoardLogger(save_dir="my_logs")
logger = pl_loggers.CSVLogger(save_dir="my_logs",name="my_csv_logs")

device = "mps" # Use 'mps' for Mac M1 or M2 Core, 'gpu' for Windows with Nvidia GPU, or 'cpu' for Windows without Nvidia GPU

trainer = pl.Trainer(max_epochs=10, accelerator=device, callbacks=[checkpoint], logger=logger)
trainer.fit(model,train_loader,val_loader)
    
trainer.test(ckpt_path="best", dataloaders=test_loader)