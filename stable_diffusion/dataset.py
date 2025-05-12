from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from text_encoder import CLIPTextEncoder

from torch.nn.utils.rnn import pad_sequence


import torch
import random

text_encoder = CLIPTextEncoder(device="cpu")


class COCOLoaderUNEt:
    def __init__(self, batch_size, img_shape, shuffle):
        self.batch_size = batch_size
        self.C = img_shape[0]
        self.H = img_shape[1]
        self.W = img_shape[2]
        self.shuffle = shuffle
        self.images_dir = '/home/cesar/Downloads/mscoco/val2017/val2017/'
        self.annotations_dir = "/home/cesar/Downloads/mscoco/annotations_trainval2017/annotations/captions_val2017.json"

        self.transforms = transforms.Compose(
                [
                    transforms.Resize((self.H, self.W), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: (x * 2) - 1)
                ]
            )
        

    def get_coco_loader(self):
        dataset = datasets.CocoCaptions(root=self.images_dir, annFile=self.annotations_dir,
                                         transform=self.transforms)
        
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=self.shuffle, num_workers=2, collate_fn=collate_fn_unet ,pin_memory=True, drop_last=True)
        
        return dataloader
    
def collate_fn_unet(batch):
    # Extract images (only)
    images = [item[0] for item in batch]  # Extract images (first element of the tuple)

    captions = [item[1] for item in batch]
    captions = ["".join(random.choice(caption_list)) for caption_list in captions]
    caption_embeddings = text_encoder(captions)
    
    # Stack the images into a single tensor (image batch)
    images = torch.stack(images, dim=0)  # Stack images along the batch dimension
    
    return images, caption_embeddings

class COCOLoader:
    def __init__(self, batch_size, img_shape, shuffle, device):
        self.batch_size = batch_size
        self.C = img_shape[0]
        self.H = img_shape[1]
        self.W = img_shape[2]
        self.shuffle = shuffle
        self.device = device
        self.images_dir = '/home/cesar/Downloads/mscoco/val2017/val2017/'
        self.annotations_dir = "/home/cesar/Downloads/mscoco/annotations_trainval2017/annotations/captions_val2017.json"


        self.transforms = transforms.Compose(
            [
                transforms.Resize((self.H, self.W), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x * 2) - 1)
            ]
        )

        self.text_encoder = CLIPTextEncoder()


    

    def get_coco_loader(self):
        dataset = datasets.CocoCaptions(root=self.images_dir, annFile=self.annotations_dir,
                                         transform=self.transforms)
        
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=self.shuffle, num_workers=2, collate_fn=collate_fn ,pin_memory=True, drop_last=True)
        
        return dataloader
        
        return dataloader
            

def collate_fn(batch):
    # Extract images (only)
    images = [item[0] for item in batch]  # Extract images (first element of the tuple)
    
    # Stack the images into a single tensor (image batch)
    images = torch.stack(images, dim=0)  # Stack images along the batch dimension
    
    return images