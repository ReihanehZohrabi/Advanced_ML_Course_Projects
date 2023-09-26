import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from model import build_model
from Dataloading import create_datasets, create_data_loaders
from utils import save_model, show_plots, SaveBestModel
import pickle

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=100,
    help='number of epochs to train our network for')
parser.add_argument('-ch', '--checkpoint_path', type=str, 
    help='checkpoint path')
parser.add_argument('-lch', '--loaded_checkpoint_path', type=str, 
    help='checkpoint path')
parser.add_argument('-prt', '--pretrained', type=int, 
    help='use pretrained model or not')
parser.add_argument('-fint', '--fine_tune_all', type=int, 
    help='fine tune all layers or not')
parser.add_argument('-btch', '--batch_size', type=int)
parser.add_argument('-exp', '--experiment', type=str)


args = vars(parser.parse_args())

# learning_parameters 
lr = 1e-2
epochs = args['epochs']
checkpoint_path = args['checkpoint_path']
loaded_checkpoint_path = args['loaded_checkpoint_path']
pretrained = args['pretrained']
fine_tune_all = args['fine_tune_all']
batch_size = args['batch_size']
experiment = args['experiment']



train_dataset, valid_dataset, test_dataset = create_datasets(experiment)

train_loader, valid_loader, _ = create_data_loaders(
    train_dataset, valid_dataset, test_dataset,batch_size,0
)



# computation device
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")

# build the model
model = build_model(
    pretrained=pretrained==1, fine_tune_all=fine_tune_all==1, num_classes=10, checkpoint_path=loaded_checkpoint_path
).to(device)
print(model)

# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.\n")

# optimizer
#optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9,weight_decay=1e-4)
# loss function
criterion = nn.CrossEntropyLoss()
# initialize SaveBestModel class
save_best_model = SaveBestModel()

# training
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # forward pass
        outputs = model(image)
        # calculate the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # backpropagation
        loss.backward()
        # update the optimizer parameters
        optimizer.step()
    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

    # validation
def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)
            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc


# lists to keep track of losses and accuracies
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []

# start the training
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")

    train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                            optimizer, criterion)
    valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,  
                                                criterion)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
    # save the best model till now if we have the least loss in the current epoch
    save_best_model(valid_epoch_loss, epoch, model, optimizer, criterion,checkpoint_path)

    if epoch % 10 == 0:

        save_model(epoch, model, optimizer, criterion,checkpoint_path)
        with open('{}train_losses.pickle'.format(checkpoint_path),'wb') as f:
                pickle.dump(train_loss, f)
        with open('{}valid_losses.pickle'.format(checkpoint_path),'wb') as f:
                pickle.dump(valid_loss, f)
        with open('{}train_accuracies.pickle'.format(checkpoint_path),'wb') as f:
                pickle.dump(train_acc, f)
        with open('{}valid_accuracies.pickle'.format(checkpoint_path),'wb') as f:
                pickle.dump(valid_acc, f)

    print('-'*50)
    

# save the loss and accuracy plots
#save_plots(train_acc, valid_acc, train_loss, valid_loss)
print('TRAINING COMPLETE')