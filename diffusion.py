import torch
import torch.nn.functional as F
from helpers import extract
import math


def linear_beta_schedule(beta_start, beta_end, timesteps):
    """
    standard linear beta/variance schedule as proposed in the original paper
    """
    return torch.linspace(beta_start, beta_end, timesteps)


# TODO: Transform into task for students
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    t = torch.linspace(0, timesteps, timesteps+1) / timesteps
    f_t = torch.cos(((t + s) / (1 + s)) * (math.pi / 2)) ** 2
    alphas_cumprod = f_t / f_t[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class Diffusion:

    def __init__(self, timesteps, get_noise_schedule, img_size):
        """
        Takes the number of noising steps, a function for generating a noise schedule as well as the image size as input.
        """
        self.timesteps = timesteps
        self.img_size = img_size

        # define beta schedule
        self.betas = get_noise_schedule(self.timesteps)

        self.alphas = 1 - self.betas
        self.alphas_cprod = torch.cumprod(self.alphas, dim= 0)
        self.alphas_cprod_p = torch.cat([torch.tensor([1]).float(), self.alphas_cprod[:-1]], 0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # TODO
        self.sqrt_alphas_cprod = torch.sqrt(self.alphas_cprod)
        self.sqrt_one_minus_alphas_cprod = torch.sqrt(1. - self.alphas_cprod)
        self.sqrt_recip_alphas=torch.sqrt(1/self.alphas)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # TODO
        self.posterior_variance = self.betas*(1-self.alphas_cprod_p) / (1-self.alphas_cprod)

    @torch.no_grad()
    def p_sample(self, model, x, t,t_index,classes=None,w=None):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean=(1+w)*model(x,t,classes)- w*model(x,t)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model_mean / sqrt_one_minus_alphas_cumprod_t
        )
        # The method should return the image at timestep t-1.
        if t_index == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + (torch.sqrt(betas_t) * noise) * sqrt_recip_alphas_t

    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3,return_final_img=False,classes=None,w=None):
        shape=(batch_size,channels,image_size,image_size)
        img = torch.rand(shape)
        imgs = []

        for t in reversed(range(0, int(self.timesteps))): 
          img= self.p_sample(model, img, torch.full((batch_size,), t),t,classes,w)
          imgs.append(img)

        stack_img = torch.stack(imgs, dim=1)                   
        return stack_img,img

    def q_sample(self, x_zero, t, noise=None):
        if noise == None:
            noise = torch.rand_like(x_zero)

        x_t = extract(self.sqrt_alphas_cprod, t, x_zero.shape) * x_zero + extract(self.sqrt_one_minus_alphas_cprod, t, x_zero.shape) * noise
        return x_t


    def p_losses(self, denoise_model, x_zero, t, noise=None, loss_type="l1",classes=None):
        if noise is None:
            noise = torch.randn_like(x_zero)
        x_noise = self.q_sample(x_zero, t, noise=noise)
        pred_noise = denoise_model(x_noise, t, classes)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, pred_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise,pred_noise)
        else:
            raise NotImplementedError()
        return loss
