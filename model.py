from typing import Any, Optional
import pytorch_lightning as pl
import torch
from torch import nn,einsum
from einops import rearrange,reduce,repeat
from functools import partial
import torch.optim as optim
from helpers import *
from diffusion import *
import config
from torchvision.utils import make_grid

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    # Note: This implements FiLM conditioning, see https://distill.pub/2018/feature-wise-transformations/ and
    # http://arxiv.org/pdf/1709.07871.pdf
    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, classes_emb_dim=None, groups=8):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(time_emb_dim or 0) + int(classes_emb_dim or 0), dim_out * 2)
        ) if exists(time_emb_dim) or exists(classes_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, class_emb=None):

        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1 1')
            scale_shift = cond_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


# Linear attention variant, scales linear with sequence length
# Shen et al.: https://arxiv.org/abs/1812.01243
# https://github.com/lucidrains/linear-attention-transformer
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


# Wu et al.: https://arxiv.org/abs/1803.08494
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class UNet(pl.LightningModule):
    def __init__(
            self,
                dim,
                timesteps,
                init_dim=None,
                out_dim=None,
                dim_mults=(1,2,4,8),
                channels=3,
                resnet_block_groups=4,
                p_uncond=0.2,
                num_classes=10,
                learning_rate=0.001
            ):
        super().__init__()
        self.p_uncond=p_uncond
        self.input_channels=channels
        self.dim=dim
        self.learning_rate=learning_rate
        self.timesteps=timesteps

        my_scheduler = (lambda x: cosine_beta_schedule(x))
        self.diffusor = Diffusion(timesteps=config.TIMESTEPS,get_noise_schedule=my_scheduler,img_size=config.IMAGE_SIZE)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(self.input_channels, init_dim, 1, padding=0)

        dims=[init_dim,*map(lambda m:self.dim * m,dim_mults)]
        in_out=list(zip(dims[:-1],dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        time_dim=dim*4

        self.time_mlp= nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.classes_emb=nn.Embedding(num_classes+1,dim)
        classes_dim=dim*4
        self.null_classes_emb=nn.Parameter(torch.randn(dim))
        self.classes_mlp = nn.Sequential(
                nn.Linear(dim, classes_dim),
                nn.GELU(),
                nn.Linear(classes_dim, classes_dim)
            )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim,classes_emb_dim = classes_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim,classes_emb_dim = classes_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim,classes_emb_dim = classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim,classes_emb_dim = classes_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim,classes_emb_dim = classes_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim,classes_emb_dim = classes_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim,classes_emb_dim = classes_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)
        self.save_hyperparameters()

    def forward(self,x,time,classes=None):
        batch=x.shape[0]
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        # TODO: Implement the class conditioning. Keep in mind that
        #  - for each element in the batch, the class embedding is replaced with the null token with a certain probability during training
        #  - during testing, you need to have control over whether the conditioning is applied or not
        #  - analogously to the time embedding, the class embedding is provided in every ResNet block as additional conditioning

        null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)
        
        if classes is not None:
            classes_emb = self.classes_emb(classes)
            condition = (classes == 10).reshape(-1,1)
            c=torch.where(condition,null_classes_emb,classes_emb)

            c = self.classes_mlp(c)
        else:
            c=self.classes_mlp(null_classes_emb)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)

            x = block2(x, t, c)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, c)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t, c)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        
        x = self.final_res_block(x, t, c)
        return self.final_conv(x)
    
    def training_step(self, batch, batch_idx):
        loss=self.common_step(batch,batch_idx)
        self.log("train_loss",loss,on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss=self.common_step(batch,batch_idx)
        self.log("val_loss",loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss=self.common_step(batch,batch_idx)
        self.log("test_loss",loss)
        return loss
    
    def common_step(self,batch,batch_idx):
        images,labels=batch
        t = torch.randint(0, self.timesteps, (len(images),))
        loss=self.diffusor.p_losses(self,images,t,loss_type="l2",classes=labels)
        return loss
    
    def on_train_epoch_end(self):
        self.eval()
        with torch.no_grad():
            self.sample_n_images(50,self.diffusor)
        self.train()
    
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(),lr=self.learning_rate,weight_decay=1e-4)
    
    def sample_n_images(self,n_images, diffusor):
        w=7
        num_classes=10
        image_classes=torch.arange(num_classes).repeat_interleave(n_images//num_classes)
        stacked_images,final_img=diffusor.sample(model=self,image_size=32,batch_size=n_images,classes=image_classes,w=w)
        #clip the image between -1 and 1
        final_img=torch.clip(final_img,-1,1)
        final_img=(final_img+1)/2
        final_img=make_grid(final_img,nrow=5)
        self.logger.experiment.add_images(f"Glyphs/{self.current_epoch}",final_img.unsqueeze(0))
