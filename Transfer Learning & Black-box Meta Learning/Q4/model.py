import torchvision.models as models
import torch.nn as nn
import torch
from ProtoNetBack import ProtoNetBack
import torch.nn.functional as F

backbone = ProtoNetBack(3)
class My_Model(nn.Module):
    def __init__(self, backbone,input_size):
        super(My_Model, self).__init__()
        self.backbone = backbone
        self.fc1 = nn.Sequential(nn.Linear(self.backbone.get_embedding_size(input_size),256),
                                           nn.ReLU(),
                                           nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(256,128),
                                           nn.ReLU(),
                                           nn.Dropout(0.5))
        self.fc3 = nn.Linear(128,10)

    def forward(self, x):
        x = self.backbone.forward(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


def build_model(pretrained=True, fine_tune_all=True, num_classes=1,checkpoint_path='./checkpoints/a/1/'):
    """
    Function to build the neural network model. Returns the model
    """
    model = My_Model(backbone=backbone,input_size=(3,32,32))
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
        checkpoint = torch.load(checkpoint_path+'best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])

        if fine_tune_all:
            print('[INFO]: Fine-tuning all layers...')
            for params in model.parameters():
                params.requires_grad = True
        elif not fine_tune_all:
            print('[INFO]: Freezing backbone layers...')
            for params in model.backbone.parameters():
                params.requires_grad = False
            
    return model