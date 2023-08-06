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




class ControlledUnetWrapperModel(torch.nn.Module):
    def __init__(self, original_model, only_mid_control=False):
        super().__init__()
        self.original_model = original_model
        self.only_mid_control = only_mid_control

    def forward(self, x, timesteps=None, context=None, *control_tensors):
        control = tuple(control_tensors)
        return self.original_model(x, timesteps, context, control, self.only_mid_control)


# prefix = '/home/player/trt2023hack'

force_export = False
force_build  = True


## 图片大小
H = 256
W = 384

onnx_dir = './onnx'
onnx_opt_dir = './onnx_opt'
engine_dir = './engine_dir'


def export_onnx(model, model_name, out_dir = onnx_dir):
    if model_name == 'control':
        x_in = torch.randn(1, 4, H//8, W//8, dtype=torch.float32).to("cuda")
        h_in = torch.randn(1, 3, H, W, dtype=torch.float32).to("cuda") ## control img
        t_in = torch.zeros(1, dtype=torch.int64).to("cuda") ## timestep
        c_in = torch.randn(1, 77, 768, dtype=torch.float32).to("cuda") ## text_emb

        ## 输入 图片 提示 timestamp 
        controls = model(x=x_in, hint=h_in, timesteps=t_in, context=c_in)

        output_names = []
        for i in range(13):
            output_names.append("out_"+ str(i))

        dynamic_table = {
            'x_in' : {0 : 'bs', 2 : 'H', 3 : 'W'}, 
            'h_in' : {0 : 'bs', 2 : '8H', 3 : '8W'}, 
            't_in' : {0 : 'bs'},
            'c_in' : {0 : 'bs'}
        }
        
        for i in range(13):
            dynamic_table[output_names[i]] = {0 : "bs"}

        torch.onnx.export(model,               
            (x_in, h_in, t_in, c_in),  
            f"./{out_dir}/{model_name}.onnx",   
            export_params=True,
            opset_version=16,
            do_constant_folding=True,
            keep_initializers_as_inputs=True,
            input_names = ['x_in', "h_in", "t_in", "c_in"], 
            output_names = output_names, 
            dynamic_axes = dynamic_table
        )

    elif model_name == 'unet':

        print('333333333333333333333333333333333333')
        x_in = torch.randn(1, 4, H//8, W//8, dtype=torch.float32).to("cuda")
        # h_in = torch.randn(1, 3, H, W, dtype=torch.float32).to("cuda") ## control img
        t_in = torch.zeros(1, dtype=torch.int64).to("cuda") ## timestep
        c_in = torch.randn(1, 77, 768, dtype=torch.float32).to("cuda") ## text_emb
        ctrl_in = []

        latent_height = H // 8
        latent_width  = W // 8
        batch_size = 1

        ctrl_in.append(torch.randn(batch_size, 320, latent_height, latent_width, dtype=torch.float32).to("cuda"))
        ctrl_in.append(torch.randn(batch_size, 320, latent_height, latent_width, dtype=torch.float32).to("cuda"))
        ctrl_in.append(torch.randn(batch_size, 320, latent_height, latent_width, dtype=torch.float32).to("cuda"))
        ctrl_in.append(torch.randn(batch_size, 320, latent_height//2, latent_width//2, dtype=torch.float32).to("cuda"))
        ctrl_in.append(torch.randn(batch_size, 640, latent_height//2, latent_width//2, dtype=torch.float32).to("cuda"))
        ctrl_in.append(torch.randn(batch_size, 640, latent_height//2, latent_width//2, dtype=torch.float32).to("cuda"))
        ctrl_in.append(torch.randn(batch_size, 640, latent_height//4, latent_width//4, dtype=torch.float32).to("cuda"))
        ctrl_in.append(torch.randn(batch_size, 1280, latent_height//4, latent_width//4, dtype=torch.float32).to("cuda"))
        ctrl_in.append(torch.randn(batch_size, 1280, latent_height//4, latent_width//4, dtype=torch.float32).to("cuda"))
        ctrl_in.append(torch.randn(batch_size, 1280, latent_height//8, latent_width//8, dtype=torch.float32).to("cuda"))
        ctrl_in.append(torch.randn(batch_size, 1280, latent_height//8, latent_width//8, dtype=torch.float32).to("cuda"))
        ctrl_in.append(torch.randn(batch_size, 1280, latent_height//8, latent_width//8, dtype=torch.float32).to("cuda"))
        ctrl_in.append(torch.randn(batch_size, 1280, latent_height//8, latent_width//8, dtype=torch.float32).to("cuda"))

        inputs = (x_in, t_in, c_in, *ctrl_in)


        wrapper_model = ControlledUnetWrapperModel(model, only_mid_control=False)

        scripted_model = torch.jit.script(wrapper_model)
        scripted_model.eval()
        
        ## 输入 图片 提示 timestamp 
        # x_out = model(x_in, 
        #     x=x_in,   # 输入latent
        #     timesteps=t_in,  ## 时间步长，会转为emb
        #     context=c_in,  # prompt emb
        #     control=ctrl_in, 
        # )

        output_names = ['x_out']
        # for i in range(13):
        #     output_names.append("out_"+ str(i)


        # (Pdb) p control[0].dtype
        # torch.float32
        # (Pdb) p control[0].shape
        # torch.Size([1, 320, 32, 48])
        # (Pdb) p control[1].shape
        # torch.Size([1, 320, 32, 48])
        # (Pdb) p control[2].shape
        # torch.Size([1, 320, 32, 48])
        # (Pdb) p control[3].shape
        # torch.Size([1, 320, 16, 24])
        # (Pdb) p control[4].shape
        # torch.Size([1, 640, 16, 24])
        # (Pdb) p control[5].shape
        # torch.Size([1, 640, 16, 24])
        # (Pdb) p control[6].shape
        # torch.Size([1, 640, 8, 12])
        # (Pdb) p control[7].shape
        # torch.Size([1, 1280, 8, 12])
        # (Pdb) p control[8].shape
        # torch.Size([1, 1280, 8, 12])
        # (Pdb) p control[9].shape
        # torch.Size([1, 1280, 4, 6])
        # (Pdb) p control[10].shape
        # torch.Size([1, 1280, 4, 6])
        # (Pdb) p control[11].shape
        # torch.Size([1, 1280, 4, 6])
        # (Pdb) p control[12].shape
        # torch.Size([1, 1280, 4, 6])
        
        

        input_names = [
            'x_in', 
            't_in',
            'c_in',
            'ctrl_in0',
            'ctrl_in1',
            'ctrl_in2',
            'ctrl_in3',
            'ctrl_in4',
            'ctrl_in5',
            'ctrl_in6',
            'ctrl_in7',
            'ctrl_in8',
            'ctrl_in9',
            'ctrl_in10',
            'ctrl_in11',
            'ctrl_in12',
        ]

        dynamic_table = {
            'x_in' : {0 : 'bs', 2 : 'H', 3 : 'W'}, 
            # 'h_in' : {0 : 'bs', 2 : '8H', 3 : '8W'}, 
            't_in' : {0 : 'bs'},
            'c_in' : {0 : 'bs'},
            'ctrl_in0': {0 : 'bs'},
            'ctrl_in1': {0 : 'bs'},
            'ctrl_in2': {0 : 'bs'},
            'ctrl_in3': {0 : 'bs'},
            'ctrl_in4': {0 : 'bs'},
            'ctrl_in5': {0 : 'bs'},
            'ctrl_in6': {0 : 'bs'},
            'ctrl_in7': {0 : 'bs'},
            'ctrl_in8': {0 : 'bs'},
            'ctrl_in9': {0 : 'bs'},
            'ctrl_in10': {0 : 'bs'},
            'ctrl_in11': {0 : 'bs'},
            'ctrl_in12': {0 : 'bs'},
            'x_out':  {0 : 'bs'},
        }


        scripted_model = torch.jit.script(wrapper_model)
        scripted_model.eval()

        torch.onnx.export(scripted_model,               
            inputs,  
            f"./{out_dir}/{model_name}.onnx",   
            export_params=True,
            opset_version=16,
            do_constant_folding=True,
            keep_initializers_as_inputs=True,
            input_names = input_names, 
            output_names = output_names, 
            dynamic_axes = dynamic_table
        )


        print('5555555555555555555555555555555555555555555555')
    


class hackathon():

    def initialize(self):
        self.apply_canny = CannyDetector()
        self.model = create_model('./models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict(f'/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.trt_logger, '')


        

        # self.model ===> cldm.cldm.ControlLDM 封装用的
        # self.model.control_model ===> cldm.cldm.ControlNet  生成control信号的ControlNet
        # self.model.model ===> ldm.models.diffusion.ddpm.DiffusionWrapper 封装用的
        # self.model.model.diffusion_model ===> cldm.cldm.ControlledUnetModel  带有control信号输入的UNET
        # self.model.first_stage_model ===> ldm.models.autoencoder.AutoencoderKL  Vae上下采样
        # self.model.cond_stage_model ===> ldm.modules.encoders.modules.FrozenCLIPEmbedder  text_encoder

        control_model = self.model.control_model
        coptrol_onnx_path = './onnx/sd_control.onnx'

        text_encoder = self.model.cond_stage_model
        text_encoder_onnx_path = './onnx/text_encoder.onnx'
        # control_model = 
        vae_model = self.model.first_stage_model
        vae_onnx_path = './onnx/vae.onnx'

        unet_model = self.model.model.diffusion_model
        unet_onnx_path = './onnx/unet.onnx'



        print(f'1111111111111111111111111111111111')
        export_onnx(model=unet_model, model_name='unet', out_dir=onnx_dir)
        print(f'12222222111111111111111111111111111111111')

        # # 获取对象的所有成员
        # all_members = dir(self.model)

        # # 分离成员对象和成员函数
        # # member_variables = [m for m in all_members if not callable(getattr(self.model, m))]
        # # member_functions = [m for m in all_members if callable(getattr(self.model, m))]
    
        # for member in all_members:
        #     if isinstance(getattr(self.model, member), torch.nn.Module):
        #         print(member)
        #     # else:
        #     #     print(f'{member}: {type(getattr(self.model, member))}')

        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        # # 展示成员对象
        # print("成员对象:")
        # for var_name in member_variables:
        #     print(var_name)

        # # 展示成员函数
        # print("\n成员函数:")
        # for func_name in member_functions:
        #     print(func_name)


        H = 256
        W = 384

        if not os.path.isfile("./engine/sd_control_fp16.engine"):
            x_in = torch.randn(1, 4, H//8, W//8, dtype=torch.float32).to("cuda")
            h_in = torch.randn(1, 3, H, W, dtype=torch.float32).to("cuda")
            t_in = torch.zeros(1, dtype=torch.int64).to("cuda")
            c_in = torch.randn(1, 77, 768, dtype=torch.float32).to("cuda")

            ## 输入 图片 提示 timestamp 
            controls = control_model(x=x_in, hint=h_in, timesteps=t_in, context=c_in)

            output_names = []
            for i in range(13):
                output_names.append("out_"+ str(i))

            dynamic_table = {'x_in' : {0 : 'bs', 2 : 'H', 3 : 'W'}, 
                             'h_in' : {0 : 'bs', 2 : '8H', 3 : '8W'}, 
                             't_in' : {0 : 'bs'},
                             'c_in' : {0 : 'bs'}}
            
            for i in range(13):
                dynamic_table[output_names[i]] = {0 : "bs"}

            torch.onnx.export(control_model,               
                                (x_in, h_in, t_in, c_in),  
                                "./onnx/sd_control_test.onnx",   
                                export_params=True,
                                opset_version=16,
                                do_constant_folding=True,
                                keep_initializers_as_inputs=True,
                                input_names = ['x_in', "h_in", "t_in", "c_in"], 
                                output_names = output_names, 
                                dynamic_axes = dynamic_table)
            
            os.system("trtexec --onnx=./onnx/sd_control_test.onnx --saveEngine=sd_control_fp16.engine --fp16 --optShapes=x_in:1x4x32x48,h_in:1x3x256x384,t_in:1,c_in:1x77x768")

        with open("./engine/sd_control_fp16.engine", 'rb') as f:
            engine_str = f.read()

        control_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_str)
        control_context = control_engine.create_execution_context()

        control_context.set_binding_shape(0, (1, 4, H // 8, W // 8))
        control_context.set_binding_shape(1, (1, 3, H, W))
        control_context.set_binding_shape(2, (1,))
        control_context.set_binding_shape(3, (1, 77, 768))
        self.model.control_context = control_context


        

        print("finished")


    def export(outdir):
        pass
        # 实现一个方法，从controlLDM导出所有model



    def process(self, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
        with torch.no_grad():
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape


            ## candy detector 生成candy图
            detected_map = self.apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            print(f'ddim_steps ========> {ddim_steps}')
            print(f'num_samples ========> {num_samples}')
            print(f'shape of control ===> {control.shape}')
            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            #此处 c_crossattn 将 prompt 和 aprompt 转为一个向量，用来操控cunet，转向量需要走 
            cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8) ## latent的维度

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=True)

            self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            
            ## cond 两部分，c_concat 是 输入的图片； c_crossattn是prompt转成的vec
            
            
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
