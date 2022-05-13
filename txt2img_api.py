import base64
from PIL import Image
import torch
import numpy as np
from tqdm.auto import tqdm, trange
from einops import rearrange
from io import BytesIO
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler


model_path = "models/ldm/text2img-large/model.ckpt"

def load_latent_model(verbose=False):
    print(f"Loading model from {model_path}")
    pl_sd = torch.load(model_path, map_location="cuda:0")
    sd = pl_sd["state_dict"]
    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model = model.half().cuda()
    model.eval()
    return model

def inference(opt, model):
    prompt = opt.prompt
    opt.ddim_eta = 0
    sampler = PLMSSampler(model)
    all_samples = []
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with model.ema_scope():
                uc = None
                if opt.scale > 0:
                    uc = model.get_learned_conditioning(opt.n_samples * [""])
                for n in trange(opt.n_iter, desc="Sampling"):
                    c = model.get_learned_conditioning(opt.n_samples * [prompt])
                    shape = [4, opt.H//8, opt.W//8]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                    conditioning=c,
                                                    batch_size=opt.n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,
                                                    eta=opt.ddim_eta)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                    for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        image = Image.fromarray(x_sample.astype(np.uint8))
                        print(type(image))
                        image.save("/home/vanga/test.png")

                        buffered = BytesIO()
                        image.save(buffered, format="JPEG")
                        image_b64 = base64.b64encode(buffered.getvalue()).decode()
                        all_samples.append(image_b64)
    return all_samples