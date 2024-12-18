################ 500 LINES OF PREVIOUS CODE TO 58 LINES OF CODE ###################

import torch
import torch.nn as nn
from torchvision.datasets import Food101
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from Pytorch_Modules import custom_zipfile_download
from Pytorch_Modules import torch_prebuilt_data_folder_format
from Pytorch_Modules import plotting
from Pytorch_Modules import datasets
from Pytorch_Modules import custom_model_builder
from Pytorch_Modules import metrics
from Pytorch_Modules import model_runtime

import os

from timeit import default_timer as timer

from torchinfo import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset=Food101(root="path",download=True) # only download if its not there, otherwise skip

#then converting the dataset into structured folder (train and test) mentioned in meta folder.
#function for re-arranging the whole dataset inside one folder to a structure of 2 folder(train, test) 
# under the parent folder.
torch_prebuilt_data_folder_format.folder_format("main_path","train.txt","test.txt","train_folder","test_folder")

#Fetching the Custom Dataset from the web.
#custom dataset zip file download from web and extract. 
custom_zipfile_download.download_extract("path","zip_file_name.zip","web_link")

plotting.plot_raw_random("path")

transform = '''T.Compose([
        T.Resize((64,64)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(10),
        T.RandomCrop(64, padding=4),
        T.TrivialAugmentWide(num_magnitude_bins=31),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])'''

plotting.plot_transformed_random("path",transform=transform)

test_transform='''T.Compose([
    T.Resize((64,64)),
    T.ToTensor(),
])'''

train_data,test_data=datasets.create_dataset(train_folder="path",test_folder="path",train_transform=transform,test_transform=test_transform,target_train_transform=None,target_test_transform=None)

train_loader,test_loader=datasets.Dataloader(train_dataset=train_data,test_dataset=test_data,batch_size="64",num_workers=os.cpu_count(), train_shuffle=True,test_shuffle=False)

train_loader_image, labels=next(iter(train_loader))
test_loader_image, labels=next(iter(test_loader))

train_loader_image.shape, test_loader_image.shape

custom_model = custom_model_builder.CustomCNN(input_shape="3",hidden_units="32",output_shape="101")

FoodCNN = models.resnet18(weights=ResNet18_Weights.DEFAULT)
FoodCNN.fc = nn.Linear("512", "101")  # Replace the final layer for 101 classes
FoodCNN = FoodCNN.to(device)

for images, labels in train_loader:
  images, labels = images.to(device), labels.to(device)
  prediction = custom_model(images)
  break
prediction[0]

loss_function = nn.CrossEntropyLoss()
opt = torch.optim.'''SGD,Adam,RMSProp......'''(custom_model.parameters(), lr="0.001", momentum="0.9")

def accuracy(output, labels):
    '''# Accuracy Function'''
    _, pred = torch.max(output, dim=1)
    return torch.sum(pred == labels).item() / len(pred) * 100

start_time = timer()

train_losses, train_accuracies, test_losses, test_accuracies = metrics.train_and_plot(
    "6",custom_model , loss_function, train_loader, test_loader, opt, metrics=accuracy)

end_time = timer()
model_runtime.run_time(start_time, end_time, device=device)

metrics.conf_matrix_for_train(model=custom_model,
                              image_path="path",
                              train_loader=train_loader)

metrics.conf_matrix_for_test(model=custom_model,
                             image_path="path",
                             train_loader=test_loader)

metrics.train_prediction(model=custom_model,
                        image_path="path",
                        )

metrics.test_prediction(model=custom_model,
                        image_path="path",
                        )

torch.save(custom_model.state_dict(), "model_name.pth")

load_model = custom_model
load_model.load_state_dict(torch.load("model_name.pth"))

batch_size="64"
summary(custom_model, input_size=(batch_size, "3", "64", "64"))

metrics.custom_image_plot(class_names_parent_path="path",
                          image_path="path",
                          device=device,
                          model=custom_model)
