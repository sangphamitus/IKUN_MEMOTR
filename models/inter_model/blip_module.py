from models.inter_model.blip.blip_itm import blip_itm
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch
model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_flickr.pth'

class BLIP_MODULE(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.image_size = 384

        self.device=device
        self.model = blip_itm(pretrained=model_url, image_size=self.image_size, vit='base')
        self.model.eval()
        self.model = self.model.to(device=device)
        self.preprocess = transforms.Compose([
        transforms.Resize((self.image_size,self.image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        self.prefix = 'a small and low quality photo of '

# model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
    def forward(self, images, caption):
        processed_image = torch.vstack([ self.preprocess(image).unsqueeze(0).to(device=self.device) for image in images])
        caption= self.prefix + caption
        itm_output =self.model(processed_image,caption,match_head='itm')
        itm_score = torch.nn.functional.softmax(itm_output,dim=1)[:,1]
        return itm_score

