################ FOOD-101 CLASSIFIER WITH RESNET #################################
################ 500 LINES OF PREVIOUS CODE TO 58 LINES OF CODE ###################

# 1. torch related libraries
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import Food101
#import torchvision.transforms as T

import torchvision.models as models
from torchvision.models import ResNet18_Weights
#from torchvision.models import EfficientNet_B0_Weights

#Modular Pytorch
from Pytorch_Modules import custom_zipfile_download
from Pytorch_Modules import torch_prebuilt_data_folder_format
from Pytorch_Modules import plotting
from Pytorch_Modules import datasets
#from Pytorch_Modules import custom_model_builder
from Pytorch_Modules import metrics
from Pytorch_Modules import model_runtime

#os
import os

#model running time
from timeit import default_timer as timer

#current model info
from torchinfo import summary

# 2. Setting Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#3. if using a prebuilt dataset from pytorch, then download using..
dataset=Food101(root="data",download=True) # only download if its not there, otherwise skip

#4. Fetching the Custom Dataset from the web. SKIP IF ALREADY DONE.
#custom dataset zip file download from web and extract. 
custom_zipfile_download.download_extract_kaggle(dataset="kmader/food41",
                                                data_path="data/food-101",
                                                zip_file_name="food41.zip")

#5. then converting the dataset into structured folder (train and test) mentioned in meta folder.
#function for re-arranging the whole dataset inside one folder to a structure of 2 folder(train, test) 
# under the parent folder. (only use this if you are using a pytorch prebuilt dataset.)
torch_prebuilt_data_folder_format.folder_format(base_dir="data/food-101/images",
                                                train_file="data/food-101/meta/train.txt",
                                                test_file="data/food-101/meta/test.txt",
                                                train_dir="data/food-101/train",
                                                test_dir="data/food-101/test")

'''#Splitting the whole dataset into train and test folders if they are not 
#(some comes with that,but some does not.)
datasets.traintest_split(input_dir="data/indian-food/indian-food",
                          output_dir="data/indian-food",
                          train_split=0.2)'''

#6. Plotting non transformed(raw) random images from the whole dataset.
plotting.plot_raw_random("data/food-101")

#7. Building the Custom CNN model

'''custom_model = custom_model_builder.CustomCNN(input_shape=3,hidden_units=32,output_shape=101)'''

#if using pretrained-model 
#model 1
ResNet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

'''# Freeze all base layers by setting requires_grad attribute to False
for param in ResNet.parameters():
    param.requires_grad = False'''

ResNet.fc = nn.Linear(512, 101)  # Replace the final layer for 101 classes
ResNet = ResNet.to(device)

'''#model 2
EfficientNet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

# Freeze all base layers by setting requires_grad attribute to False
for param in ResNet.parameters():
    param.requires_grad = False

EfficientNet.fc=nn.Linear(1408,101)
EfficientNet=EfficientNet.to(device)'''

#8. Setup pretrained weights (plenty of these available in torchvision.models)
resnet_weights = torchvision.models.ResNet18_Weights.DEFAULT

# Get transforms from weights (these are the transforms that were used to obtain the weights)
resnet_transforms = resnet_weights.transforms() 
print(f"Automatically created transforms: {resnet_transforms}")

'''#8. Setup pretrained weights (plenty of these available in torchvision.models)
efficientnet_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT

# Get transforms from weights (these are the transforms that were used to obtain the weights)
efficientnet_transforms = efficientnet_weights.transforms() 
print(f"Automatically created transforms: {efficientnet_transforms}")'''

#9. current model info
batch_size=64
summary(model=ResNet, 
        input_size=(batch_size, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])

'''batch_size=64
summary(model=EfficientNet, 
        input_size=(batch_size, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])'''

# 10. Transforming and plotting the same raw images.
plotting.plot_transformed_random("data/food-101",transform=resnet_transforms)
#plotting.plot_transformed_random("data2/pizza_steak_sushi",transform=efficientnet_transforms)

#11 .Now creating the format for training and testing dataset in order to upload to the dataloader.

resnet_train_data,resnet_test_data=datasets.create_dataset(train_folder="data/food-101/train",
                                             test_folder="data/food-101/test",

                                             train_transform=resnet_transforms,
                                             test_transform=resnet_transforms,
                                             
                                             target_train_transform=None,
                                             target_test_transform=None)

'''efficientnet_train_data,efficientnet_test_data=datasets.create_dataset(train_folder="data2/pizza_steak_sushi/train",
                                             test_folder="data2/pizza_steak_sushi/test",

                                             train_transform=efficientnet_transforms,
                                             test_transform=efficientnet_transforms,
                                             
                                             target_train_transform=None,
                                             target_test_transform=None)'''

#12. Preparing Dataloader
resnet_train_loader,resnet_test_loader=datasets.Dataloader(train_dataset=resnet_train_data,
                                             test_dataset=resnet_test_data,

                                             batch_size=64,
                                             num_workers=os.cpu_count(),
                                             
                                             train_shuffle=True,
                                             test_shuffle=False)

'''efficientnet_train_loader,efficientnet_test_loader=datasets.Dataloader(train_dataset=efficientnet_train_data,
                                             test_dataset=efficientnet_test_data,

                                             batch_size=64,
                                             num_workers=os.cpu_count(),
                                             
                                             train_shuffle=True,
                                             test_shuffle=False)'''

#13. Untrained Prediction
torch.manual_seed(41)
for images, labels in resnet_train_loader:
  images, labels = images.to(device), labels.to(device)
  prediction = ResNet(images)
  break
prediction[0]

'''#13. Untrained Prediction
torch.manual_seed(40)
for images, labels in efficientnet_train_loader:
  images, labels = images.to(device), labels.to(device)
  prediction = EfficientNet(images)
  break
prediction[0]'''

#14. Loss Function and Optimizer
loss_function = nn.CrossEntropyLoss()

resnet_opt1 = torch.optim.Adam(ResNet.parameters(), lr=0.001)
resnet_opt2= torch.optim.SGD(ResNet.parameters(),lr=0.001,momentum=0.9)

'''efficientnet_opt1 = torch.optim.Adam(EfficientNet.parameters(), lr=0.001)
efficientnet_opt2= torch.optim.SGD(EfficientNet.parameters(),lr=0.001,momentum=0.9)'''

#15. accuracy function
def accuracy(output, labels):
    '''# Accuracy Function'''
    _, pred = torch.max(output, dim=1)
    return torch.sum(pred == labels).item() / len(pred) * 100

#16. Training the Model
start_time = timer()

experiment_configs = [
    {
        'model': ResNet,  # First model
        'optimizer': resnet_opt2,
        'epochs': 15,  # Number of epochs for model1
        'name': 'ResNet_Food_Classifying_Exp2'
    }
]

# Define the DataLoader for each model (train and test)
train_loaders = [resnet_train_loader]
test_loaders = [resnet_test_loader]

# Call the training function
metrics.train_plot_tensorboard_multiple_experiments(experiment_configs, train_loaders, test_loaders, loss_function)

end_time = timer()
model_runtime.run_time(start_time, end_time, device=device)

#17. confusion matrix for both train and test
metrics.conf_matrix_for_train(model=ResNet,
                              image_path="data/food-101/train",
                              train_loader=resnet_train_loader)

metrics.conf_matrix_for_test(model=ResNet,
                             image_path="data/food-101/test",
                             test_loader=resnet_test_loader)

#18. Train and Test images prediction
metrics.train_prediction(class_names_parent_path="data/food-101/train",model=ResNet,
                        image_path="data/food-101/train",
                        )

metrics.test_prediction(class_names_parent_path="data/food-101/test",model=ResNet,
                        image_path="data/food-101/test",
                        )

#19. Saving and Loading Model
torch.save(ResNet.state_dict(), "Food-101_Classifier.pth")
load_model = ResNet
load_model.load_state_dict(torch.load("Food-101_Classifier.pth"))

#20. Testing the custom image
metrics.custom_image_plot(class_names_parent_path="data/food-101/test",
                          image_path="data/food-101/pizza.jpg",
                          device=device,
                          model=ResNet)
