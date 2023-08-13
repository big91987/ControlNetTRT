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
from cuda import cudart
import datetime


def torch_dtype_from_trt(trt_dtype):
    # Convert TensorRT dtype to PyTorch dtype
    return {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32:   torch.int32,
        # Add other dtype conversions as needed
    }[trt_dtype]


def make_cuda_graph(engine_path, model_name, trt_logger):

    assert os.path.exists(engine_path)

    with open(engine_path, 'rb') as f:
        engine_str = f.read()

    engine = trt.Runtime(trt_logger).deserialize_cuda_engine(engine_str)
    context = engine.create_execution_context()

    # Get IO information
    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)
    input_names = [x for x in lTensorName if engine.get_tensor_mode(x) == trt.TensorIOMode.INPUT]
    output_names = [x for x in lTensorName if engine.get_tensor_mode(x) == trt.TensorIOMode.OUTPUT]
    input_shapes = [engine.get_tensor_shape(x) for x in input_names]
    output_shapes = [engine.get_tensor_shape(x) for x in output_names]
    input_dtypes = [engine.get_tensor_dtype(x) for x in input_names]
    output_dtypes = [engine.get_tensor_dtype(x) for x in output_names]

    # Create CUDA stream
    _, stream = cudart.cudaStreamCreate()

    # Initialize buffers using PyTorch tensors on CUDA
    buffer = []
    for i, i_shape in enumerate(input_shapes):
        i_dtype = torch_dtype_from_trt(input_dtypes[i])
        i_shape = tuple(i_shape)
        i_data = torch.empty(size=i_shape, dtype=i_dtype, device='cuda')
        context.set_binding_shape(i, i_shape)
        buffer.append(i_data)

    for i, o_shape in enumerate(output_shapes):
        o_dtype = torch_dtype_from_trt(output_dtypes[i])
        o_shape = tuple(o_shape)
        o_data = torch.empty(size=o_shape, dtype=o_dtype, device='cuda')
        buffer.append(o_data)

    # # Set binding shapes
    for i, input_name in enumerate(input_names):
        context.set_binding_shape(i, input_shapes[i])

    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], int(buffer[i].data_ptr()))
    
    context.execute_async_v2([int(x.data_ptr()) for x in buffer], stream)

    # Capture CUDA graph
    cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
    context.execute_async_v2([int(x.data_ptr()) for x in buffer], stream)
    _, graph = cudart.cudaStreamEndCapture(stream)
    _, graphExe = cudart.cudaGraphInstantiate(graph, 0)  # for CUDA >= 12

    return buffer, graphExe, stream, context, lTensorName


class hackathon():

    def initialize(self):
        self.apply_canny = CannyDetector()
        self.model = create_model('./models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.trt_logger, '')

        uc1b_plan = './engine/uc1b.engine'
        self.model.uc1b_buff, self.model.uc1b_ge, self.model.uc1b_stream, self.model.uc1b_context, self.model.uc1b_bnames = \
        make_cuda_graph(uc1b_plan, model_name='uc1b', trt_logger=self.trt_logger)

        vd_plan = './engine/vd.engine'
        self.model.vd_buff, self.model.vd_ge, self.model.vd_stream, self.model.vd_context, \
                self.model.vd_bnames = make_cuda_graph(vd_plan, model_name='vd', trt_logger=self.trt_logger)
        
        clip_plan = './engine/clip.engine'
        self.model.clip_buff, self.model.clip_ge, self.model.clip_stream, self.model.clip_context, \
                self.model.clip_bnames = make_cuda_graph(clip_plan, model_name='clip', trt_logger=self.trt_logger)
        
        print(" --------- warm up begin -----------")
        path = "./warmup/bird_0.jpg"
        img = cv2.imread(path)
        start = datetime.datetime.now().timestamp()
        new_img = self.process(img,
                "a bird", 
                "best quality, extremely detailed", 
                "longbody, lowres, bad anatomy, bad hands, missing fingers", 
                1, 
                256, 
                20,
                False, 
                1, 
                9, 
                2946901, 
                0.0, 
                100, 
                200)
        end = datetime.datetime.now().timestamp()
        print(f" --------- warm up end, dt = {(end-start)*1000} ms-----------")
        print("finished")


    def __del__(self):
        pass


    def process(self, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
        ddim_steps=8
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
