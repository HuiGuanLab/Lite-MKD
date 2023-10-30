
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from gradcam import GradCAM, GradCAMpp
from gradcam.utils import visualize_cam
from torchvision import transforms
import PIL
import matplotlib.pyplot as plt
 
 
def main():
    resnet = torchvision.models.resnet50(pretrained=True)
    resnet.eval()
    gradcam = GradCAM.from_config(model_type='resnet', arch=resnet, layer_name='layer4')
 
    # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    pil_img = PIL.Image.open('imp_datasets/video_datasets/data/kinetics/rgb_l8/air_drumming/-eGnPDG5Pxs_000053_000063/00000001.jpg')
    torch_img = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])(pil_img).to('cpu')
    normed_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]
 
    # get a GradCAM saliency map on the class index 10.
    mask, logit = gradcam(normed_img, class_idx=None)
 
    # make heatmap from mask and synthesize saliency map using heatmap and img
    heatmap, cam_result = visualize_cam(mask, torch_img)
 
    plt.figure()
    heatmap =  transforms.ToPILImage()(heatmap)
    heatmap.save('heatmap.jpg')
    

    plt.figure()
    cam_result = transforms.ToPILImage()(cam_result)
    cam_result.save('cam_result.jpg')
    
 
 
if __name__ == '__main__':
    main()