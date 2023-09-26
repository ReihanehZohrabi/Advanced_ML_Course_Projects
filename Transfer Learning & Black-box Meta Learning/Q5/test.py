import torch
from tqdm.auto import tqdm
from model import build_model
from Dataloading import create_datasets, create_data_loaders
import argparse



folders = ["Alphabet_of_the_Magi","Anglo-Saxon_Futhorc","Arcadian","Armenian","Asomtavruli_(Georgian)","Balinese","Bengali","Blackfoot_(Canadian_Aboriginal_Syllabics)","Braille","Burmese_(Myanmar)","Cyrillic","Early_Aramaic",
           "Futurama","Grantha","Greek","Gujarati","Hebrew","Inuktitut_(Canadian_Aboriginal_Syllabics)","Japanese_(hiragana)","Japanese_(katakana)","Korean","Latin","Malay_(Jawi_-_Arabic)","Mkhedruli_(Georgian)",
           "N_Ko","Ojibwe_(Canadian_Aboriginal_Syllabics)","Sanskrit","Syriac_(Estrangelo)","Tagalog","Tifinagh"]


parser = argparse.ArgumentParser()

parser.add_argument('-ch', '--checkpoint_path', type=str, 
    help='checkpoint path')
parser.add_argument('-exp', '--experiment', type=str)
parser.add_argument('-lst', '--last_checkpoint', type=int)

parser.add_argument('-btch', '--batch_size', type=int)


args = vars(parser.parse_args())

checkpoint_path = args['checkpoint_path']
experiment = args['experiment']
batch_size = args['batch_size']



device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")



train_dataset, test_dataset,image_datasets = create_datasets()

train_loader, test_loader,train_set_sizes, test_set_sizes,class_names,classes  = create_data_loaders(
    train_dataset, test_dataset,image_datasets,batch_size,0
)


def test(model, testloader,Multi=False,Folder=''):
    """
    Function to test the model
    """
    model.eval()
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in (enumerate(testloader)):
            counter += 1
            image, labels = data
            if Multi:
              image[0] = image[0].to(device)

              labels = labels.to(device)
            else: 
              image = image.to(device)

            if Multi:
                outputs = model(image,folders.index(folder))
            else:
                outputs = model(image)
            
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
          
    final_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return final_acc

def test_best_model(model, checkpoint, test_loader,Multi,folder):
    model.load_state_dict(checkpoint['model_state_dict'])
    test_acc = test(model, test_loader,Multi,folder)
    print(f"Task {folder} Accuracy: {test_acc:.3f}")


if experiment == 'a':

  for folder in folders:
    chkpth=checkpoint_path+'a/'+folder+'/'
    model = build_model(
                experiment,num_classes=len(class_names[folder]), checkpoint_path=chkpth
                ).to(device)
    try:
      best_model_cp = torch.load(chkpth+'best_model.pth')
    except:
      best_model_cp = torch.load(chkpth+'model_epoch20.pth')

    print(f"Task {folder}:")
    test_best_model(model, best_model_cp, test_loader[folder],False,'')
else:
    if experiment == 'b_1':
        chkpth=checkpoint_path+'b/1/'
    else:
        chkpth=checkpoint_path+'b/2/'
    model = build_model(
                experiment,num_classes=classes, checkpoint_path=chkpth
                ).to(device)
    best_model_cp = torch.load(chkpth+'best_model.pth')

    for folder in folders:
      test_best_model(model, best_model_cp, test_loader[folder],True,folder)

      