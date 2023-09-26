import torch
import torchvision
from torchvision import transforms,datasets
import random
from torch.utils.data import random_split,Subset
import copy
import numpy as np

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor()
    ])


folders = ["Alphabet_of_the_Magi","Anglo-Saxon_Futhorc","Arcadian","Armenian","Asomtavruli_(Georgian)","Balinese","Bengali","Blackfoot_(Canadian_Aboriginal_Syllabics)","Braille","Burmese_(Myanmar)","Cyrillic","Early_Aramaic",
           "Futurama","Grantha","Greek","Gujarati","Hebrew","Inuktitut_(Canadian_Aboriginal_Syllabics)","Japanese_(hiragana)","Japanese_(katakana)","Korean","Latin","Malay_(Jawi_-_Arabic)","Mkhedruli_(Georgian)",
           "N_Ko","Ojibwe_(Canadian_Aboriginal_Syllabics)","Sanskrit","Syriac_(Estrangelo)","Tagalog","Tifinagh"]


def mm(loader):
  for item in loader:
    yield item 

def get_multi_ds(loaders):
  g = [mm(loaders[f]) for f in folders]
  flags = [0 for i in range (len (loaders))]
  while sum(flags)<len(loaders):
    for i in range (len(loaders)):
      if (flags[i]==0):
        try:
          x , y = g[i].__next__()
          yield [x, torch.tensor(i)], y
        except StopIteration:
          flags[i]=1

def create_datasets():
    """
    Function to build the training, validation, and testing dataset.
    """

    data_dir = "./Omniglot/"

    paths = [data_dir + folder for folder in folders]
    image_datasets = {x: datasets.ImageFolder(data_dir+x,transform) for x in folders}
    train_set = copy.deepcopy(image_datasets)
    test_set = copy.deepcopy(image_datasets)

    for folder in folders:
      for i in range(int(len(train_set[folder])//20)):
        del train_set[folder].imgs[14*i:14*i+6]
        del test_set[folder].imgs[6*i+6:6*i+20]
      
    return train_set, test_set,image_datasets



def create_data_loaders(train_set, test_set,image_datasets,BATCH_SIZE,NUM_WORKERS):
    """
    Function to build the data loaders.
    """
    train_loader = {x: torch.utils.data.DataLoader(train_set[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
              for x in folders}
    test_loader = {x: torch.utils.data.DataLoader(test_set[x], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
              for x in folders}
    train_set_sizes = {x: len(train_set[x]) for x in folders}
    test_set_sizes = {x: len(test_set[x]) for x in folders}
    class_names = {x: image_datasets[x].classes for x in folders}
    classes = {x: len(image_datasets[x].classes) for x in folders}
    return train_loader, test_loader,train_set_sizes, test_set_sizes,class_names,classes



