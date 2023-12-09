from model import UNet
from diffusion import Diffusion,linear_beta_schedule,cosine_beta_schedule
import torch
import local_config as config
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision.utils import save_image
import torchvision.utils as vutils
from torchvision.utils import make_grid

def sample_n_images(n_images, diffusor, model,store_path):
    w=7
    num_classes=10
    image_classes=torch.arange(num_classes).repeat_interleave(n_images//num_classes)
    stacked_images,final_img=diffusor.sample(model=model,image_size=32,batch_size=n_images,classes=image_classes,w=w)
    #clip the image between -1 and 1
    final_img=torch.clip(final_img,-1,1)
    #final_img=(final_img+1)/2
    final_img=make_grid(final_img,nrow=5,normalize=True)
    final_img = final_img.cpu().numpy()
    plt.imsave(store_path+"all.png", np.transpose(final_img, (1, 2, 0)))

if __name__=="__main__":
    model=UNet.load_from_checkpoint(config.CKPT_DIR_PATH+"/model_epoch(epoch=29)_valloss(val_loss=0.06).ckpt")
    n_images=50
    my_scheduler = (lambda x: cosine_beta_schedule(x))
    diffusor=Diffusion(timesteps=config.TIMESTEPS,get_noise_schedule=my_scheduler,img_size=config.IMAGE_SIZE)
    sample_n_images(n_images, diffusor, model, config.GEN_IMG_PATH)
    