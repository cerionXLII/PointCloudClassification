#This is a simple container of the models


import numpy as np
import torch.nn as nn
import torch

class ModelContainer(nn.Module):
    def __init__(self):
        super(ModelContainer, self).__init__()
        self.Ensemble = [] #Where we keep an ensemble of the models
      
    def addModel(self, model):
        self.Ensemble.append(model)

    def to(self, device):
        for model in self.Ensemble:
            model.to(device)
        return self
    
    def eval(self):
        for model in self.Ensemble:
            model.eval()
        return self
    
    def train(self):
        for model in self.Ensemble:
            model.train()
        return self

    def forward(self, x):
        preds = [model(x) for model in self.Ensemble]
        ensemblePred = torch.mean(torch.stack(preds), dim=0)
        return ensemblePred
    

