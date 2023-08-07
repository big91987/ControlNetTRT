from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import os
import tensorrt as trt

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from ldm.util import log_txt_as_img, exists, instantiate_from_config



def make_context(engine_path, model_name, trt_logger):
    H = 256
    W = 384
    assert os.path.exists(engine_path)
    latent_height = H // 8
    latent_width  = W // 8
    latent_channel  = 4
    img_hegint = H
    img_width = W
    img_channel = 3
    embedding_dim = 768
    text_maxlen = 77
    batch_size = 1

    if model_name == 'control':
        input_shapes = {
            'x_in' : [batch_size, latent_channel, latent_height, latent_width],
            'h_in' : [batch_size, img_channel, img_hegint, img_width], 
            't_in' : [batch_size], 
            'c_in' : [batch_size, text_maxlen, embedding_dim],
        }

    elif model_name == 'unet':

        input_shapes = {
            'x_in' : [batch_size, latent_channel, latent_height, latent_width], 
            't_in' : [batch_size], 
            'c_in' : [batch_size, text_maxlen, embedding_dim], 
            'ctrl0': [batch_size, 320, latent_height, latent_width],
            'ctrl1': [batch_size, 320, latent_height, latent_width],
            'ctrl2': [batch_size, 320, latent_height, latent_width],
            'ctrl3': [batch_size, 320, latent_height//2, latent_width//2],
            'ctrl4': [batch_size, 640, latent_height//2, latent_width//2],
            'ctrl5': [batch_size, 640, latent_height//2, latent_width//2],
            'ctrl6': [batch_size, 640, latent_height//4, latent_width//4],
            'ctrl7': [batch_size, 1280, latent_height//4, latent_width//4],
            'ctrl8': [batch_size, 1280, latent_height//4, latent_width//4],
            'ctrl9': [batch_size, 1280, latent_height//8, latent_width//8],
            'ctrl10': [batch_size, 1280, latent_height//8, latent_width//8],
            'ctrl11': [batch_size, 1280, latent_height//8, latent_width//8],
            'ctrl12': [batch_size, 1280, latent_height//8, latent_width//8],
        }
    else:
        return None

    with open(engine_path, 'rb') as f:
        engine_str = f.read()

    engine = trt.Runtime(trt_logger).deserialize_cuda_engine(engine_str)
    context = engine.create_execution_context()
    
    for i, input_name in enumerate(input_shapes.keys()):
        context.set_binding_shape(i, input_shapes[input_name])
    
    return context



class hackathon():

    def initialize(self):
        self.apply_canny = CannyDetector()
        self.model = create_model('./models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.trt_logger, '')


        H = 256
        W = 384

        control_onnx = './onnx/control/model.onnx'
        control_plan = './engine/control.engine'


        clip_onnx = './onnx/clip/model.onnx'
        clip_plan = './engine/clip.engine'

        vae_onnx = './onnx/vae/model.onnx'
        vae_plan = './engine/vae.engine'

        unet_onnx = './onnx/unet/model.onnx'
        unet_plan = './engine/unet.engine'

        self.model.control_context = make_context(control_plan, model_name='control', trt_logger=self.trt_logger)

        self.model.unet_context = make_context(unet_plan, model_name='unet', trt_logger=self.trt_logger)


        print("finished")



    def process(self, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
        
        # ddim_steps = 10
        with torch.no_grad():
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            detected_map = self.apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=True)

            self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = self.ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(num_samples)]
        return results
