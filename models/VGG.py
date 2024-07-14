import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.hub import load_state_dict_from_url

pretrained_model_urls = {'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'}

class VGG(nn.Module):
     def __init__(self, n_class=2):
        super(VGG, self).__init__()
        model_pre = models.vgg16()
        model_pre.load_state_dict(load_state_dict_from_url(pretrained_model_urls['vgg16'], progress=True))

        for param in model_pre.features.parameters():
            param.required_grad = False

        num_features = model_pre.classifier[6].in_features
        features = list(model_pre.classifier.children())[:-1] 
        features.extend([nn.Linear(num_features, n_class)])
        self.classifier = nn.Sequential(*features) 
        model_pre.classifier = self.classifier
        self.model = model_pre

     def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)
        return x