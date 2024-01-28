import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import math
from datasets import load_dataset
from torchmetrics import Accuracy, AUROC
import pandas as pd
import numpy as np

path_config = {
    "folder": "PATH_CONTAIN_MODEL/", 
    "model": "models/bert-base-chinese",
    "data": "data/afqmc"
}

class AFQTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(path_config['folder'] + path_config['model'])
        self.linear = nn.Linear(self.base_model.config.hidden_size, 1)
        self.act = nn.Sigmoid()

        # initialize
        self.linear.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param.data)
                else:
                    nn.init.normal_(param.data)
        else:
            pass
    
    def forward(self, X):
        ret = self.base_model(**X)
        logits = self.act(self.linear(ret.pooler_output)).view(-1)
        return logits

class AFQData(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(path_config['folder'] + path_config['model'])
        self.batch_size = config['batch_size']

    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            self.train_set, self.dev_set = load_dataset(
                path_config['folder'] + path_config['data'], 
                data_files={"train": "train.json", "validate": "dev.json"}, split=['train', 'validate']
            )

    def preprocess(self, batch_sample):
        sentence1 = []
        sentence2 = []
        labels = []
        idx = []
        for sample in batch_sample:
            sentence1.append(sample['sentence1'])
            sentence2.append(sample['sentence2'])
            labels.append(float(sample['label']))
            idx.append(sample['idx'])
        X = self.tokenizer(sentence1, sentence2, return_tensors='pt', padding="longest", truncation=True, max_length=512)
        y = torch.tensor(labels)
        idx = torch.tensor(idx)
        return X, y, idx
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, collate_fn=self.preprocess, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dev_set, batch_size=self.batch_size, collate_fn=self.preprocess)


class AFQModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AFQTorch()
        self.loss_fct = nn.BCELoss(reduction='none')
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.acc = Accuracy(task='binary')
        self.auc = AUROC(task='binary')
        
    def process_batch(self, batch, batch_idx, stage='train'):
        X, y, idx = batch
        preds = self.model(X)
        loss = self.loss_fct(preds, y)
        if stage == 'train':
            loss = torch.mean(loss)
        return loss, preds, y, idx

    def training_step(self, batch, batch_idx):
        opts = self.optimizers()
        schs = self.lr_schedulers()
        opts.zero_grad()

        loss, _ , _ ,_ = self.process_batch(batch, batch_idx)

        self.log("train/loss", loss)
        self.log("train/lr", schs.get_last_lr()[0])

        self.manual_backward(loss)

        opts.step()
        schs.step()
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels, idx = self.process_batch(batch, batch_idx, 'val')
        self.acc.update(preds, labels)
        self.auc.update(preds, labels)

        self.log("val/loss_median", torch.median(loss), sync_dist=True)
        self.log("val/loss_mean", torch.mean(loss), sync_dist=True)

        # [4, batch_size]
        return torch.stack([loss, preds, labels, idx])

    def validation_epoch_end(self, validation_step_outputs):
        final_acc = self.acc.compute()
        final_auc = self.auc.compute()
        self.acc.reset()
        self.auc.reset()
        self.log("val/acc", final_acc, sync_dist=True)
        self.log("val/auc", final_auc, sync_dist=True)

        all_val_out = self.all_gather(validation_step_outputs)
        if self.trainer.is_global_zero:
            result = torch.cat(all_val_out, dim=-1).cpu().detach()
            result = result.permute([1, 0, 2]).reshape((4, -1))
            df = pd.DataFrame({"idx": result[3].int(), "loss": result[0], "pred": result[1], "labels": result[2].int()})
            df.to_csv(path_config['folder'] + "out/out_{epoch}.csv".format(epoch = self.trainer.current_epoch))
        self.trainer.strategy.barrier()

    def configure_optimizers(self):
        lr = self.config['lr']
        weight_decay = self.config['weight_decay']

        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if 'LayerNorm' in name or 'bias' in name:
                no_decay.append(param)
            else:
                decay.append(param)
        
        optimizer = torch.optim.AdamW(
            params= [
                {"params": decay, "lr": lr, "weight_decay": weight_decay},
                {"params": no_decay, "lr": lr, "weight_decay": 0.0},
            ]
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            anneal_strategy='linear',
            max_lr=lr, 
            total_steps=self.trainer.estimated_stepping_batches
        )
        scheduler = {"scheduler": scheduler, "interval": "step"}
        
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()


if __name__ == "__main__":
    config = {
        "batch_size": 32,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "epoch": 10
    }
    data = AFQData(config)
    data.setup('fit')
    loader = data.val_dataloader()
    print(next(iter(loader)))
