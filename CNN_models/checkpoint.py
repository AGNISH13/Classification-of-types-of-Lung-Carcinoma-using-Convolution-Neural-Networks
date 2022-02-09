

from CNN_models import densenet121, googlenet, resnet18, vgg16

def check_point(folder_path, CNN_type):

    if CNN_type == 'googlenet':
        googlenet.g_net(folder_path)
    elif CNN_type == 'resnet18':
        resnet18.res_net(folder_path)
    elif CNN_type == 'vgg16':
        vgg16.vgg_net(folder_path)
    elif CNN_type == 'densenet121':
        densenet121.dense_net(folder_path)
    else:
        #Default CNN model
        googlenet.g_net(folder_path) 

        
        
        
