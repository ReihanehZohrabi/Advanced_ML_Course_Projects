
import torchvision.models as models
import torch.nn as nn
import torch
from ProtoNetBack import ProtoNetBack
import torch.nn.functional as F

backbone = ProtoNetBack(3)

folders = ["Alphabet_of_the_Magi","Anglo-Saxon_Futhorc","Arcadian","Armenian","Asomtavruli_(Georgian)","Balinese","Bengali","Blackfoot_(Canadian_Aboriginal_Syllabics)","Braille","Burmese_(Myanmar)","Cyrillic","Early_Aramaic",
           "Futurama","Grantha","Greek","Gujarati","Hebrew","Inuktitut_(Canadian_Aboriginal_Syllabics)","Japanese_(hiragana)","Japanese_(katakana)","Korean","Latin","Malay_(Jawi_-_Arabic)","Mkhedruli_(Georgian)",
           "N_Ko","Ojibwe_(Canadian_Aboriginal_Syllabics)","Sanskrit","Syriac_(Estrangelo)","Tagalog","Tifinagh"]


class Model1(nn.Module):
    def __init__(self, backbone,input_size,num_classes):
        super(Model1, self).__init__()
        self.backbone = backbone
        self.fc1 = nn.Sequential(nn.Linear(self.backbone.get_embedding_size(input_size),256),
                                           nn.ReLU(),
                                           nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(256,128),
                                           nn.ReLU(),
                                           nn.Dropout(0.5))
        self.fc3 = nn.Linear(128,num_classes)

    def forward(self, x):
        x = self.backbone.forward(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

class Model2_1(nn.Module):
    def __init__(self,backbone,input_size,foldersizes):
        super(Model2_1, self).__init__()
        self.backbone = backbone
        self.fc1 = nn.Sequential(nn.Linear(self.backbone.get_embedding_size(input_size),256),
                                           nn.ReLU(),
                                           nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(256,128),
                                           nn.ReLU(),
                                           nn.Dropout(0.5))
        self.layer3 = nn.ModuleList([nn.Linear(128, foldersizes[folder]) for folder in folders])


    def forward(self, x, index):
        x = self.backbone.forward(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.layer3[index](x)
        output = F.log_softmax(x, dim=1)
        return output

class Model2_2(nn.Module):
    def __init__(self,backbone,input_size,foldersizes):
        super(Model2_2, self).__init__()
        self.backbone = backbone
        self.layer1 = nn.Sequential(nn.Linear(self.backbone.get_embedding_size(input_size),256),
                                           nn.ReLU(),
                                           nn.Dropout(0.5))
     

        self.layer2 = nn.ModuleList([nn.Sequential(nn.Linear(256,128),
                                           nn.ReLU(),
                                           nn.Dropout(0.5),
                                           nn.Linear(128,foldersizes[folder])) for folder in folders])

      
    def forward(self, x, index):
        x = self.backbone.forward(x)
        x = torch.flatten(x, 1)
        x = self.layer1(x)
        x = self.layer2[index](x)
        output = F.log_softmax(x, dim=1)
        return output



def build_model(experiment, num_classes=1,checkpoint_path='./checkpoints/',pretrained=False):
    """
    Function to build the neural network model. Returns the model
    """

    if experiment == 'a':
        model = Model1(backbone=backbone,input_size=(3,32,32),num_classes=num_classes)
    elif experiment == 'b_1':
        model = Model2_1(backbone=backbone,input_size=(3,32,32),foldersizes=num_classes)
    else:
        model = Model2_2(backbone=backbone,input_size=(3,32,32),foldersizes=num_classes)
      
    return model