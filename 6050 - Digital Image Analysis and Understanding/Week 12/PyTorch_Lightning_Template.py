import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self,filename):
        super(MyDataset, self).__init__()
#         TODO: implement what happens when someone calls: dataset = MyDataset()
#         Pull in relevant file information, images lists, etc.
    
    def __getitem__(self, idx):
#         TODO: implement what happens when someone calls dataset[idx]
#         Return an image and it's associated label at location idx 
    
    def __len__(self):
#         TODO: implement what happens when someone calls len(dataset)
#         Determine the number of images in the dataset

train_dataset = MyDataset("training.txt")
validation_dataset = MyDataset("validation.txt")
test_dataset = MyDataset("test.txt")

train_loader = DataLoader(train_dataset,batch_size=1)
val_loader = DataLoader(validation_dataset,batch_size=1)
test_loader = DataLoader(test_dataset,batch_size=1)



class LitNetwork(pl.LightningModule):
    def __init__(self):
        super(LitNetwork, self).__init__()

        # TODO: setup network here
        
        n_classes = 10 #Change num_classes to the number of classification categories in your dataset
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.val_acc = torchmetrics.Accuracy("multiclass",num_classes=n_classes,average='micro')
        self.test_acc = torchmetrics.Accuracy("multiclass",num_classes=n_classes,average='micro')

    def forward(self, x):
        # TODO: perform the forward pass, which happens when someone calls network(x)
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
logger = pl_loggers.TensorBoardLogger(save_dir="my_logs")
#logger = pl_loggers.CSVLogger(save_dir="my_logs",name="my_csv_logs")

device = "mps" # Use 'mps' for Mac M1 or M2 Core, 'gpu' for Windows with Nvidia GPU, or 'cpu' for Windows without Nvidia GPU

trainer = pl.Trainer(max_epochs=10, accelerator=device, callbacks=[checkpoint], logger=logger)
trainer.fit(model,train_loader,val_loader)
    
trainer.test(ckpt_path="best", dataloaders=test_loader)