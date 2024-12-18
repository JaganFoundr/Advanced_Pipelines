################ FOOD-101 CLASSIFIER WITH RESNET #################################
################ 500 LINES OF PREVIOUS CODE TO 58 LINES OF CODE ###################

# 1. torch related libraries
import torch
import torch.nn as nn
from torchvision.datasets import Food101
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import ResNet18_Weights

#Modular Pytorch
from Pytorch_Modules import custom_zipfile_download
from Pytorch_Modules import torch_prebuilt_data_folder_format
from Pytorch_Modules import plotting
from Pytorch_Modules import datasets
from Pytorch_Modules import custom_model_builder
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

'''#3. if using a prebuilt dataset from pytorch, then download using..
dataset=Food101(root="data",download=True) # only download if its not there, otherwise skip'''

#4. Fetching the Custom Dataset from the web. SKIP IF ALREADY DONE.
#custom dataset zip file download from web and extract. 
custom_zipfile_download.download_extract_kaggle(dataset="kmader/food41",
                                                data_path="data/food-101",
                                                zip_file_name="food41.zip")

#5. then converting the dataset into structured folder (train and test) mentioned in meta folder.
#function for re-arranging the whole dataset inside one folder to a structure of 2 folder(train, test) 
# under the parent folder. (only use this if you are using a pytorch prebuilt dataset.)
torch_prebuilt_data_folder_format.folder_format(base_dir="data/food-101/images",
                                                train_file="data/food-101/meta/meta/train.txt",
                                                test_file="data/food-101/meta/meta/test.txt",
                                                train_dir="data/food-101/train",
                                                test_dir="data/food-101/test")

'''#Splitting the whole dataset into train and test folders if they are not 
#(some comes with that,but some does not.)
datasets.traintest_split(input_dir="data/indian-food/indian-food",
                          output_dir="data/indian-food",
                          train_split=0.2)'''

#6. Plotting non transformed(raw) random images from the whole dataset.
plotting.plot_raw_random("data/food-101")

#7. Transforming and plotting the same raw images.

transform = T.Compose([
        T.Resize((224,224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(10),
        T.RandomCrop(224, padding=4),
        T.TrivialAugmentWide(num_magnitude_bins=31),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

plotting.plot_transformed_random("path",transform=transform)

#8 .Now creating the format for training and testing dataset in order to upload to the dataloader.
test_transform=T.Compose([
    T.Resize((64,64)),
    T.ToTensor(),
])

train_data,test_data=datasets.create_dataset(train_folder="data/food-101/train",
                                             test_folder="data/food-101/test",

                                             train_transform=transform,
                                             test_transform=test_transform,
                                             
                                             target_train_transform=None,
                                             target_test_transform=None)

#9. Preparing Dataloader
train_loader,test_loader=datasets.Dataloader(train_dataset=train_data,
                                             test_dataset=test_data,

                                             batch_size=64,
                                             num_workers=os.cpu_count(),
                                             
                                             train_shuffle=True,
                                             test_shuffle=False)

#10. shape of the images in the train loader
train_loader_image, labels=next(iter(train_loader))
test_loader_image, labels=next(iter(test_loader))

train_loader_image.shape, test_loader_image.shape

#11. Building the Custom CNN model

custom_model = custom_model_builder.CustomCNN(input_shape=3,hidden_units=32,output_shape=101)

#if using pretrained-model
ResNet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
ResNet.fc = nn.Linear(512, 101)  # Replace the final layer for 101 classes
ResNet = ResNet.to(device)

#12. Untrained Prediction
for images, labels in train_loader:
  images, labels = images.to(device), labels.to(device)
  prediction = ResNet(images)
  break
prediction[0]

#13. Loss Function and Optimizer
loss_function = nn.CrossEntropyLoss()
opt = torch.optim.Adam(ResNet.parameters(), lr=0.001)

#14. accuracy function
def accuracy(output, labels):
    '''# Accuracy Function'''
    _, pred = torch.max(output, dim=1)
    return torch.sum(pred == labels).item() / len(pred) * 100

#15. Training the Model
start_time = timer()

train_losses, train_accuracies, test_losses, test_accuracies = metrics.train_and_plot(
    13, ResNet , loss_function, train_loader, test_loader, opt, metrics=accuracy)

end_time = timer()
model_runtime.run_time(start_time, end_time, device=device)

#16. confusion matrix for both train and test
metrics.conf_matrix_for_train(model=ResNet,
                              image_path="data/food-101/train",
                              train_loader=train_loader)

metrics.conf_matrix_for_test(model=ResNet,
                             image_path="data/food-101/test",
                             train_loader=test_loader)

#17. Train and Test images prediction
metrics.train_prediction(model=ResNet,
                        image_path="data/food-101/train",
                        )

metrics.test_prediction(model=ResNet,
                        image_path="data/food-101/test",
                        )

#18. Saving and Loading Model
torch.save(ResNet.state_dict(), "Food_Classifier.pth")

load_model = ResNet
load_model.load_state_dict(torch.load("Food_Classifier.pth"))

#19. current model info
batch_size=64
summary(custom_model, input_size=(batch_size, 3, 224, 224))

#20. Testing the custom image
metrics.custom_image_plot(class_names_parent_path="data/food-101/train",
                          image_path="data/food-101/pizza.jpg",
                          device=device,
                          model=ResNet)
