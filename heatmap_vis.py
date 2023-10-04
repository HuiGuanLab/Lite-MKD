
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image,deprocess_image,preprocess_image
from torchvision import transforms
import PIL
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
 
def main():
    
    resnet = torchvision.models.resnet18(pretrained=True)  

    
    target_layer = [resnet.layer4] 
    target_category = None
    resnet.eval() 
    
    gradcam = GradCAM(model=resnet, target_layers=target_layer, use_cuda=False)  
    
    
    # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    
    # pil_img = PIL.Image.open('/home/zty/204/data/kinetics/l8/rgb_l8/air_drumming/-eGnPDG5Pxs_000053_000063/00000001.jpg')
    # torch_img = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])(pil_img).to('cpu')
    # normed_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]
    
    image_path='/home/zty/204/data/kinetics/l8/rgb_l8/air_drumming/-eGnPDG5Pxs_000053_000063/00000001.jpg'
    rgb_img = cv2.imread(image_path, 1) 
    rgb_img = np.float32(rgb_img) / 255

    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]) 

    
    
 
    # get a GradCAM saliency map on the class index 10.
    
    grayscale_cam = gradcam(input_tensor=input_tensor)
    
    grayscale_cam = grayscale_cam[0]
    visualization = show_cam_on_image(rgb_img, grayscale_cam)
    cv2.imwrite(f'cam_dog.jpg', visualization)
    
    
 
 
if __name__ == '__main__':
    main()