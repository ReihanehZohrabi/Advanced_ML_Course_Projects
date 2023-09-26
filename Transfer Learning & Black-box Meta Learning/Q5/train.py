
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from model import build_model
from Dataloading import create_datasets, create_data_loaders
from utils import save_model, show_plots, SaveBestModel
import pickle
import os
import numpy as np

folders = ["Alphabet_of_the_Magi","Anglo-Saxon_Futhorc","Arcadian","Armenian","Asomtavruli_(Georgian)","Balinese","Bengali","Blackfoot_(Canadian_Aboriginal_Syllabics)","Braille","Burmese_(Myanmar)","Cyrillic","Early_Aramaic",
           "Futurama","Grantha","Greek","Gujarati","Hebrew","Inuktitut_(Canadian_Aboriginal_Syllabics)","Japanese_(hiragana)","Japanese_(katakana)","Korean","Latin","Malay_(Jawi_-_Arabic)","Mkhedruli_(Georgian)",
           "N_Ko","Ojibwe_(Canadian_Aboriginal_Syllabics)","Sanskrit","Syriac_(Estrangelo)","Tagalog","Tifinagh"]


# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=100,
    help='number of epochs to train our network for')
parser.add_argument('-ch', '--checkpoint_path', type=str, 
    help='checkpoint path')
parser.add_argument('-btch', '--batch_size', type=int)
parser.add_argument('-exp', '--experiment', type=str)


args = vars(parser.parse_args())

# learning_parameters 
lr = 1e-3
epochs = args['epochs']
checkpoint_path = args['checkpoint_path']
batch_size = args['batch_size']
experiment = args['experiment']



train_dataset, test_dataset,image_datasets = create_datasets()

train_loader, test_loader,train_set_sizes, test_set_sizes,class_names,classes  = create_data_loaders(
    train_dataset, test_dataset,image_datasets,batch_size,0
)



device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")


criterion = nn.CrossEntropyLoss()

def train(model, trainloader, optimizer, criterion,Multi=False):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    totals = 0
    total =0
    corrects=np.zeros((30))
    Falses=np.zeros((30))
    total_tasks= np.zeros((30))
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        if Multi:
          image[0] = image[0].to(device)
        else: 
          image = image.to(device)

        labels = labels.to(device)
        total_task = len(labels)
        optimizer.zero_grad()
        if Multi:
                outputs = model(image[0],image[1])
        else:
                outputs = model(image)
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        running_loss_task = loss.item()
        _, preds = torch.max(outputs.data, 1)
        
        total += labels.shape[0]
        totals = labels.shape[0]
        train_running_correct += (preds == labels).sum().item()
        running_correct_task = (preds == labels).sum().item()
        # if Multi:
        #   task = np.unique(image[1])
        #   corrects[task[0]]+= running_correct_task
        #   Falses[task[0]] += running_loss_task
        #   total_tasks[task[0]] += totals
        loss.backward()
        optimizer.step()
      
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct /(total))
    # for i in range(30):
    #   print(f'Task: {i} Acc: {np.sum(corrects[i])/(total_tasks[i])} loss: {np.sum(Falses[i])/(total_tasks[i])}')
    return epoch_loss, epoch_acc

    # validation
def validate(model, testloader, criterion,Multi=False):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    total = 0
    totals=0
    corrects=np.zeros((30))
    Falses=np.zeros((30))
    total_tasks= np.zeros((30))
    with torch.no_grad():
      for i, data in tqdm(enumerate(testloader), total=len(testloader)):
        counter += 1
        image, labels = data
        if Multi:
          image[0] = image[0].to(device)
        else: 
          image = image.to(device)

        labels = labels.to(device)
        total_task = len(labels)
        if Multi:
                outputs = model(image[0],image[1])
        else:
                outputs = model(image)
        loss = criterion(outputs, labels)
        valid_running_loss += loss.item()
        running_loss_task = loss.item()
        _, preds = torch.max(outputs.data, 1)
        
        total += labels.shape[0]
        totals = labels.shape[0]
        valid_running_correct += (preds == labels).sum().item()
        running_correct_task = (preds == labels).sum().item()
        if Multi:
          task = np.unique(image[1])
          corrects[task[0]]+= running_correct_task
          Falses[task[0]] += running_loss_task
          total_tasks[task[0]] += totals
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct /(total))
    if experiment != 'a':
        for i in range(30):
          print(f'Task: {i} \n Test Acc: {np.sum(corrects[i])/(total_tasks[i])} \n loss: {np.sum(Falses[i])/(total_tasks[i])}')
    return epoch_loss, epoch_acc



def generator(loader):
  for item in loader:
    yield item 

def get_multi_ds(loaders):
  g = [generator(ldr) for ldr in loaders]
  flags = [0 for i in range (len (loaders))]
  while sum(flags)<len(loaders):
    for i in range (len(loaders)):
      if (flags[i]==0):
        try:
          x , y = g[i].__next__()
          yield [x, torch.tensor(i)], y
        except StopIteration:
          flags[i]=1


# lists to keep track of losses and accuracies
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []
if experiment == 'a':

    for folder in folders:
        save_best_model = SaveBestModel()
        if not os.path.exists(checkpoint_path+'a/'+folder):
            os.makedirs(checkpoint_path+'a/'+folder)
        chkpth=checkpoint_path+'a/'+folder+'/'
        model = build_model(
                experiment,num_classes=len(class_names[folder]), checkpoint_path=chkpth
                ).to(device)
                
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        
        # total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.\n")

        print(f"[INFO]: Training Task {folder}:")
        for epoch in range(epochs):
            print(f"[INFO]: Epoch {epoch+1} of {epochs}")

            train_epoch_loss, train_epoch_acc = train(model, train_loader[folder], 
                                                    optimizer, criterion)
            valid_epoch_loss, valid_epoch_acc = validate(model, test_loader[folder],  
                                                        criterion)
            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)
            train_acc.append(train_epoch_acc)
            valid_acc.append(valid_epoch_acc)
            print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
            print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
            # save the best model till now if we have the least loss in the current epoch
            save_best_model(valid_epoch_loss, epoch, model, optimizer, criterion,chkpth)

            if epoch+1 % 10 == 0:

                save_model(epoch, model, optimizer, criterion,chkpth)
                with open('{}train_losses.pickle'.format(chkpth),'wb') as f:
                        pickle.dump(train_loss, f)
                with open('{}valid_losses.pickle'.format(chkpth),'wb') as f:
                        pickle.dump(valid_loss, f)
                with open('{}train_accuracies.pickle'.format(chkpth),'wb') as f:
                        pickle.dump(train_acc, f)
                with open('{}valid_accuracies.pickle'.format(chkpth),'wb') as f:
                        pickle.dump(valid_acc, f)

            print('-'*50)
        print('*'*50)   
else:
    dataloader_train = torch.load('multi_task_batches')
    dataloader_test = torch.load('multi_task_batches_test')

    if experiment == 'b_1':
        chkpth=checkpoint_path+'b/1/'
    else:
        chkpth=checkpoint_path+'b/2/'
    model = build_model(
                experiment,num_classes=classes, checkpoint_path=chkpth
                ).to(device)
      
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    save_best_model = SaveBestModel()
    

        # total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.\n")

    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")

        train_epoch_loss, train_epoch_acc = train(model, dataloader_train, 
                                                    optimizer, criterion,True)
        valid_epoch_loss, valid_epoch_acc = validate(model, dataloader_test,  
                                                        criterion,True)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
            # save the best model till now if we have the least loss in the current epoch
        save_best_model(valid_epoch_loss, epoch, model, optimizer, criterion,chkpth)

        if epoch+1 % 10 == 0:

            save_model(epoch, model, optimizer, criterion,chkpth)
            with open('{}train_losses.pickle'.format(chkpth),'wb') as f:
                        pickle.dump(train_loss, f)
            with open('{}valid_losses.pickle'.format(chkpth),'wb') as f:
                        pickle.dump(valid_loss, f)
            with open('{}train_accuracies.pickle'.format(chkpth),'wb') as f:
                        pickle.dump(train_acc, f)
            with open('{}valid_accuracies.pickle'.format(chkpth),'wb') as f:
                        pickle.dump(valid_acc, f)

            print('-'*50)
        print('*'*50)  


