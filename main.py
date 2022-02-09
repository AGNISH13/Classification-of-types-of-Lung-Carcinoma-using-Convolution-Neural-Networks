

from CNN_models import checkpoint
import argparse


parser = argparse.ArgumentParser(description='Training and Evaluating different CNN models')
#Paths
parser.add_argument('-data', '--data_folder',
                     type=str, default='Data_Folder/')
parser.add_argument('-cnn', '--CNN_type',
                     type=str, default='googlenet')



args = parser.parse_args()
folder_path = args.data_folder
CNN_model = args.CNN_type



checkpoint.check_point(folder_path, CNN_model)
print('\n\nYour task is complete!')


