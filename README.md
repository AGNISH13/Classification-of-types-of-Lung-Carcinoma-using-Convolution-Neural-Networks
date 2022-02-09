# Classification-of-types-of-Lung-Carcinoma-using-Convolution-Neural-Networks

## Project Description
This is a python-based project classifying the type of Lung Carcinoma disease based on deep analysis of histopathological image samples through the application of four popular Convolution Neural Networks, namely `GoogLeNet`, `VGG-16`, `ResNet-18` and `DenseNet-121`, one at a time, giving a different Training and Validation accuracy for each epoch.

## Classes of Division
In this project, the histopathological image samples of human lungs have been classified into three categories, namely:  
- `Lung Benign tissue`  
- `Lung Adenocarcinoma`  
- `Lung Squamous cell carcinoma`

## Convolution Neural Network models used
Four CNN models have been applied on the dataset, namely:  
-	`GoogLeNet`  
-	`Visual Geometry Group (VGG-16)`  
-	`ResNet-18`  
-	`DenseNet-121`

## Train-Validation Learning Curve
Train-Validation Curve is a popular method to helps us confirm normal behavioural characteristics of model over increasing number of epochs 
 
All the models have been trained over `20` epochs with batch_size of `60`
-     GoogleNet
     ![image](https://user-images.githubusercontent.com/89198752/153136792-b68cb600-5f30-4ddc-bb78-3dee08e0e2f9.png)
-     VGG-16
     ![image](https://user-images.githubusercontent.com/89198752/153138623-6c81103e-471b-46a2-9524-a8abc846dd9e.png)
-     ResNet-18
     ![image](https://user-images.githubusercontent.com/89198752/153137163-08121fd9-d5c4-4e68-8b4d-483fb7876bbe.png)
-     DenseNet-121
     ![image](https://user-images.githubusercontent.com/89198752/153137273-eb8d7c7b-1747-4c7a-b117-7528121ccc9b.png)

## Dependencies
Since the entire project is based on `Python` programming language, it is necessary to have Python installed in the system. It is recommended to use Python with version `>=3.6`.
The Python packages which are in use in this project are  `matplotlib`, `numpy`, `pandas`, `torch` and `torchvision`. All these dependencies can be installed just by the following command line argument
- pip install `requirements.txt`

## Code implementation
- ### Data paths :
      Current directory ----> Data_Folder
                                  |
                                  |
                                  |               
                                  ------------------>  train
                                  |                      |
                                  |             --------------------
                                  |             |        |         |
                                  |             V        V         V
                                  |           lung_n  lung_aca  lung_scc
                                  |
                                  |
                                  |              
                                  ------------------>   val
                                                         |
                                                --------------------
                                                |        |         |
                                                V        V         V
                                              lung_n  lung_aca  lung_scc
                                              
                               
- Where the folders `train` and `val` contain the folders `lung_n`, `lung_aca`and `lung_scc`, which include the original histopathological images of respective type of lung carcinoma in `.jpg`/`.png` format.
- `Note:` The folders `lung_n`, `lung_aca`and `lung_scc` contain the histopathological images of 'Benign', 'Adenocarcinoma' and 'Squamous cell carcinoma' samples respectively.

- ### Training and Evaluating different CNN models :
      -help

      optional arguments:
        -h, --help            show this help message and exit
        -data DATA_FOLDER, --data_folder DATA_FOLDER
        -cnn CNN_TYPE, --CNN_type CNN_TYPE
        
-  ### Run the following for training and validation :
  
      `python main.py -data Data_Folder -cnn vgg16`
      
-  ### Specific tokens :

          GoogLeNet: 'googlenet'
          VGG-16: 'vgg16'
          ResNet-18: 'resnet18'
          DenseNet-121: 'densenet121'
