from PIL import Image
import torch
from models.visualize import Visualizer
from utils.box_ops import box_cxcywh_to_xyxy

def cut_image_from_bbox( image, boxes):
    left = torch.clamp(boxes[:, 0], 0).tolist()
    top = torch.clamp(boxes[:, 1], 0).tolist()
    right = torch.clamp(boxes[:, 2], 0).tolist()
    bottom = torch.clamp(boxes[:, 3], 0).tolist()
    cropped_image =[]
    
    for i in range(len(left)):
        img=image.copy().crop((left[i], top[i], right[i], bottom[i]))
        cropped_image.append(img)

    return cropped_image

class INTER_MODEL():
    def __init__(self, model, device,with_visualizer=False):
        self.model = model
        self.device = device
        self.visualizer = Visualizer(save_dir="D:\Thesis\DamnShit\Hello\MeMOTR_IKUN\Visual")   
        self.with_visualizer=with_visualizer
    def predict(self, caption, new_bbox,width,height,temp_img,old_images=[]):
        boxes = box_cxcywh_to_xyxy(new_bbox)
        boxes = (boxes * torch.as_tensor([width,height, width, height], dtype=torch.float).to(boxes.device))
    
        crop_image=cut_image_from_bbox(temp_img,boxes)
        crop_image2=crop_image+ old_images

        masks = []
        if len(crop_image2)>0:
            probs = self.model(crop_image2,caption)
            if self.with_visualizer:
                self.visualizer(temp_img,boxes,caption,probs)
            masks = probs
        return masks,crop_image