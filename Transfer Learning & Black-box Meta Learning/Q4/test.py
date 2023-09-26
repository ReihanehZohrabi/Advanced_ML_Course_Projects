import torch
from tqdm.auto import tqdm
from model import build_model
from Dataloading import create_datasets, create_data_loaders
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('-ch', '--checkpoint_path', type=str, 
    help='checkpoint path')
parser.add_argument('-exp', '--experiment', type=str)
parser.add_argument('-lst', '--last_checkpoint', type=int)



args = vars(parser.parse_args())

checkpoint_path = args['checkpoint_path']
experiment = args['experiment']
last_checkpoint = args['last_checkpoint']



# computation device
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")


model = build_model(
    pretrained=False, fine_tune_all=False, num_classes=10
).to(device)
# load the best model checkpoint
best_model_cp = torch.load(checkpoint_path+'best_model.pth')
best_model_epoch = best_model_cp['epoch']
print(f"Best model was saved at {best_model_epoch} epochs\n")

# load the last model checkpoint
last_model_cp = torch.load(checkpoint_path+'model_epoch'+str(last_checkpoint)+'.pth')
last_model_epoch = last_model_cp['epoch']
print(f"Last model was saved at {last_model_epoch} epochs\n")

# get the test dataset and the test data loader
train_dataset, valid_dataset, test_dataset = create_datasets(experiment)
_, _, test_loader = create_data_loaders(
    train_dataset, valid_dataset, test_dataset,32,0
)


def test(model, testloader):
    """
    Function to test the model
    """
    # set model to evaluation mode
    model.eval()
    print('Testing')
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
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # loss and accuracy for the complete epoch
    final_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return final_acc


# test the last epoch saved model
def test_last_model(model, checkpoint, test_loader):
    print('Loading last epoch saved model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_acc = test(model, test_loader)
    print(f"Last epoch saved model accuracy: {test_acc:.3f}")


# test the best epoch saved model
def test_best_model(model, checkpoint, test_loader):
    print('Loading best epoch saved model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_acc = test(model, test_loader)
    print(f"Best epoch saved model accuracy: {test_acc:.3f}")

if __name__ == '__main__':
    test_last_model(model, last_model_cp, test_loader)
    test_best_model(model, best_model_cp, test_loader)