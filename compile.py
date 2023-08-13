from share import *
import config
import time
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import os
import tensorrt as trt
import uuid
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from ldm.util import log_txt_as_img, exists, instantiate_from_config
import onnx
import shutil
import onnx_graphsurgeon as gs

from onnxruntime.transformers.float16 import convert_float_to_float16


def onnx_export(model, model_name, inputs, input_names, output_names, dynamic_axes, out_path, convert_fp16 = True, const_folding = True, input_shapes = {}):
    
    out_dir = os.path.dirname(out_path)
    assert os.path.exists(out_dir)
    
    tmp_dir = os.path.join(out_dir, f'__tmp_"{uuid.uuid1()}__"')
    tmp_path = os.path.join(tmp_dir, 'model.onnx')
    os.makedirs(tmp_dir, exist_ok=False)

    if input_shapes:
        torch.onnx.export(
            model,               
            inputs,  
            tmp_path,   
            export_params=True,
            opset_version=16,
            do_constant_folding=True,
            keep_initializers_as_inputs=True,
            input_names = input_names, 
            output_names = output_names, 
            dynamic_axes = dynamic_axes,
            input_shapes = input_shapes,
        )
    else:
        torch.onnx.export(
            model,               
            inputs,  
            tmp_path,   
            export_params=True,
            opset_version=16,
            do_constant_folding=True,
            keep_initializers_as_inputs=True,
            input_names = input_names, 
            output_names = output_names, 
            dynamic_axes = dynamic_axes
        )


    tmp_model = onnx.load(tmp_path)

    # for clip change -inf to -100000
    # if model_name == 'clip':
    #     for node in tmp_model.graph.node:
    #         # if node.name == "/text_model/ConstantOfShape_1":
    #         if node.op_type == "ConstantOfShape":
    #             print(node)
    #             attr = node.attribute[0]
    #             print(attr)
    #             if attr.name == "value" and attr.t.data_type == onnx.TensorProto.FLOAT:
    #                 np_array = np.frombuffer(attr.t.raw_data, dtype=np.float32).copy()
    #                 print("raw array", np_array)
    #                 np_array[np_array == -np.inf] = -100000  # 将所有负无穷的值改为-100000
    #                 attr.t.raw_data = np_array.tobytes() 
    #                 print("new array", np_array) 


    if convert_fp16:
        onnx.shape_inference.infer_shapes_path(tmp_path)
        tmp_model = convert_float_to_float16(
            tmp_model, keep_io_types=True, disable_shape_infer=True
        )

    if const_folding:
        graph = gs.import_onnx(tmp_model)
        graph.fold_constants().cleanup()
        tmp_model = gs.export_onnx(graph)
    
    onnx.save_model(
        tmp_model,
        out_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"weight.pb",
        convert_attribute=False,
    )

    shutil.rmtree(tmp_dir)
    

# class UnetWithSpecControl(torch.nn.Module):
#     def __init__(self, unet_model, control_model, only_mid_control=False):
#         super().__init__()
#         self.unet_model = unet_model
#         self.control_model = control_model
#         self.only_mid_control = only_mid_control
#         # self.control_scale = control_scale

#     def forward(self, x, h, t, c):
#         control = self.control_model(x, h, t, c)
#         return self.unet_model(x, t, c, control, self.only_mid_control)




# import torch

class UnetWithSpecControl(torch.nn.Module):

    def __init__(self, unet_model, control_model, only_mid_control=False):
        super().__init__()
        self.unet_model = unet_model
        self.control_model = control_model
        self.only_mid_control = only_mid_control

    def forward(self, x, h, t, c, control_scale):
        
        control = self.control_model(x, h, t, c)
        
        # 对control的13个元素依次乘以系数 
        for i in range(13):
            control[i] *= control_scale[i]
            
        return self.unet_model(x, t, c, control, self.only_mid_control)
    


class UC1B(torch.nn.Module):

    def __init__(self, unet_model, control_model, only_mid_control=False):
        super().__init__()
        self.unet_model = unet_model
        self.control_model = control_model
        self.only_mid_control = only_mid_control

    def forward(self, x, t, h_t, h_u, c_t, c_u, control_scale, un_scale):
        
        ## 列表 需要在外面转, 不允许携带字典，list，标量输入，请转成tensor

        x_cat = torch.cat([x, x], 0)
        h_cat = torch.cat([h_t, h_u], 0)
        c_cat = torch.cat([c_t, c_u], 0)

        control_cat = self.control_model(x_cat, h_cat, t, c_cat)
        # 对control的13个元素依次乘以系数 
        for i in range(13):
            control_cat[i] *= control_scale[i]
        m_cat = self.unet_model(x_cat, t, c_cat, control_cat, self.only_mid_control)
        model_t = m_cat[0].unsqueeze(0)
        model_u = m_cat[1].unsqueeze(0)
        return model_u + un_scale * (model_t - model_u)  
         
    



# class DDIM_(torch.nn.Module):
#     def __init__(self, unet_model, control_model, only_mid_control=False):
#         super().__init__()

#         # model_t = self.model.apply_model(x, t, c)
#         # model_uncond = self.model.apply_model(x, t, unconditional_conditioning)
#         # model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

#         self.unet_model = unet_model
#         self.control_model = control_model
#         self.only_mid_control = only_mid_control

#     def forward(self, x, t, c_t, c_u, h_t, h_u,  
#             cs0, cs1, cs2, cs3, cs4, cs5, cs6, cs7, cs8, cs9, cs10, cs11, cs12,
#             un_scale
#         ):
#         # h_t = torch.cat(c_concat, 1)
#         # h_u = torch.cat(uc_concat, 1)
#         # c_t = torch.cat(c_crossattn, 1)
#         # c_u = torch.cat(uc_crossattn, 1)

#         control_scale = (cs0, cs1, cs2, cs3, cs4, cs5, cs6, cs7, cs8, cs9, cs10, cs11, cs12)

#         control_t = self.control_model(x, h_t, t, c_t)
#         control_t = [control_scale[i] * control_t[i] for i in range(13)]
#         model_t = self.unet_model(x, t, c_t, control_t, self.only_mid_control)
#         control_u = self.control_model(x, h_u, t, c_u)
#         control_u = [control_scale[i] * control_u[i] for i in range(13)]
#         model_u = self.unet_model(x, t, c_u, control_u, self.only_mid_control)

#         return model_u + un_scale * (model_t - model_u)



# class DDIM_(torch.nn.Module):
#     def __init__(self, unet_model, control_model, only_mid_control=False):
#         super().__init__()
        
#         self.unet_model = unet_model
#         self.control_model = control_model
#         self.only_mid_control = only_mid_control

#     def forward(self, 
#                 x, t, 
#                 c_t, c_u, 
#                 h_t, h_u,
#                 un_scale):
        
#         control_t = self.control_model(x, h_t, t, c_t)
#         control_u = self.control_model(x, h_u, t, c_u)

#         for i in range(13):
#             control_t[i] = control_scale[i] * control_t[i]
#             control_u[i] = control_scale[i] * control_u[i]
        
#         model_t = self.unet_model(x, t, c_t, control_t, self.only_mid_control)
#         model_u = self.unet_model(x, t, c_u, control_u, self.only_mid_control)
        
#         return model_u + un_scale * (model_t - model_u)
    


# class DDIM_(torch.nn.Module):
#     def __init__(self, unet_model, control_model, only_mid_control=False):
#         super().__init__()
        
#         self.unet_model = unet_model
#         self.control_model = control_model
#         self.only_mid_control = only_mid_control
        

#     def forward(self, 
#                 x, t, 
#                 c_t, c_u, 
#                 h_t, h_u,
#                 un_scale,
#                 ):
        
#         control_t = self.control_model(x, h_t, t, c_t)
#         control_u = self.control_model(x, h_u, t, c_u)

#         # for i in range(13):
#         #     control_t[i] = strength * control_t[i]
#         #     control_u[i] = strength * control_u[i]
        
#         model_t = self.unet_model(x, t, c_t, control_t, self.only_mid_control)
#         model_u = self.unet_model(x, t, c_u, control_u, self.only_mid_control)
        
#         return model_u + un_scale * (model_t - model_u)



class ControlledUnetWrapperModel(torch.nn.Module):
    def __init__(self, original_model, only_mid_control=False):
        super().__init__()
        self.original_model = original_model
        self.only_mid_control = only_mid_control

    def forward(self, x, t, c, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12):
        control = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12]
        return self.original_model(x, t, c, control, self.only_mid_control)
    
# from ldm.modules.diffusionmodules.model import Encoder, Decoder
class VD(torch.nn.Module):
    def __init__(self, decoder, post_quant_conv):
        super().__init__()
        self.decoder = decoder
        self.post_quant_conv = post_quant_conv

    def forward(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
    

class ClipWrapper(torch.nn.Module):
    def __init__(self, trans, layer="last", layer_idx=None):
        super().__init__()
        self.trans = trans
        self.layer = layer
        self.layer_idx = layer_idx

    def forward(self, tokens):
        outputs = self.trans(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z


## 图片大小
H = 256
W = 384

def export(model, model_name, out_path, convert_fp16=True, const_folding=True, batch_size = 1):

    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    input_shapes = {}
    if model_name == 'control':
        x_in = torch.randn(batch_size, 4, H//8, W//8, dtype=torch.float32).to("cuda")
        h_in = torch.randn(batch_size, 3, H, W, dtype=torch.float32).to("cuda") ## control img
        t_in = torch.zeros(batch_size, dtype=torch.int64).to("cuda") ## timestep
        c_in = torch.randn(batch_size, 77, 768, dtype=torch.float32).to("cuda") ## text_emb

        inputs = (x_in, h_in, t_in, c_in)

        input_names = ['x_in', 'h_in', 't_in', 'c_in']
        ## 输入 图片 提示 timestamp 
        # controls = model(*inputs)

        output_names = []
        for i in range(13):
            output_names.append("ctrl"+ str(i))

        dynamic_table = {
            'x_in' : {0 : 'bs', 2 : 'H', 3 : 'W'}, 
            'h_in' : {0 : 'bs', 2 : '8H', 3 : '8W'}, 
            't_in' : {0 : 'bs'},
            'c_in' : {0 : 'bs'}
        }
        
        for i in range(13):
            dynamic_table[output_names[i]] = {0 : "bs"}


    # elif model_name == 'ddim':
        
    #     x = torch.randn(batch_size, 4, H//8, W//8, dtype=torch.float32).to("cuda")
    #     t = torch.zeros(1, dtype=torch.int64).to("cuda") ## timestep

    #     c_t = torch.randn(batch_size, 77, 768, dtype=torch.float32).to("cuda") ## text_emb
    #     c_u = torch.randn(batch_size, 77, 768, dtype=torch.float32).to("cuda") ## text_emb
    #     h_t = torch.randn(batch_size, 3, H, W, dtype=torch.float32).to("cuda") ## control img
    #     h_u = torch.randn(batch_size, 3, H, W, dtype=torch.float32).to("cuda") ## control img

       
    #     # strength = 1.0
    #     un_scale = 1.0
        

    #     inputs = (x, t, c_t, c_u, h_t, h_u, un_scale)

    #     input_names = [
    #         'x', 't', 
    #         'c_t', 'c_u', 'h_t', 'h_u', 
    #         'un_scale',
    #     ]
    #     ## 输入 图片 提示 timestamp 
    #     # controls = model(*inputs)

    #     output_names = ['x_out']
        
    #     dynamic_table = {
    #         'x' : {0 : 'bs', 2 : 'lh', 3 : 'lw'},
    #         'c_t' : {0 : 'bs'},
    #         'c_u' : {0 : 'bs'},
    #         'h_t' : {0 : 'bs', 2 : 'H', 3 : 'W'}, 
    #         'h_u' : {0 : 'bs', 2 : 'H', 3 : 'W'}, 
    #         'x_out': {0 : 'bs', 2 : 'lh', 3 : 'lw'},
    #     }

    #     # input_shapes: {
    #     #     "control_scale": [len(control_scale)],

    #     # }
            
    elif model_name == 'unet_with_spec_control':

        # trtexec --onnx=./onnx2/unet_with_spec_control/model_pruned.onnx --saveEngine=./engine2/uc2b_preuned.engine --fp16 --optShapes=x_in:2x4x32x48,h_in:2x3x256x384,t_in:1,c_in:2x77x768,control_scale:13
        
        x_in = torch.randn(batch_size, 4, H//8, W//8, dtype=torch.float32).to("cuda")
        h_in = torch.randn(batch_size, 3, H, W, dtype=torch.float32).to("cuda") ## control img
        t_in = torch.zeros(1, dtype=torch.int64).to("cuda") ## timestep
        c_in = torch.randn(batch_size, 77, 768, dtype=torch.float32).to("cuda") ## text_emb
        control_scale = torch.randn([13], dtype=torch.float32).to("cuda")

        inputs = (x_in, h_in, t_in, c_in, control_scale)

        input_names = ['x_in', 'h_in', 't_in', 'c_in', 'control_scale']
        ## 输入 图片 提示 timestamp 
        # controls = model(*inputs)

        output_names = ['x_out']
        
        dynamic_table = {
            'x_in' : {0 : 'bs', 2 : 'H', 3 : 'W'}, 
            'h_in' : {0 : 'bs', 2 : '8H', 3 : '8W'}, 
            # 't_in' : {0 : 'bs'},
            'c_in' : {0 : 'bs'},
            'x_out': {0 : 'bs', 2 : 'H', 3 : 'W'},
        }

    elif model_name == 'uc1b':
        # x, t, h_t, h_u, c_t, c_u, control_scale, un_scale
        x = torch.randn(batch_size, 4, H//8, W//8, dtype=torch.float32).to("cuda")
        t = torch.zeros(1, dtype=torch.int64).to("cuda") ## timestep
        h_t = torch.randn(batch_size, 3, H, W, dtype=torch.float32).to("cuda") ## control img
        h_u = torch.randn(batch_size, 3, H, W, dtype=torch.float32).to("cuda") ## control img
        
        c_t = torch.randn(batch_size, 77, 768, dtype=torch.float32).to("cuda") ## text_emb
        c_u = torch.randn(batch_size, 77, 768, dtype=torch.float32).to("cuda") ## text_emb
        control_scale = torch.randn([13], dtype=torch.float32).to("cuda")
        un_scale = torch.randn([1], dtype=torch.float32).to("cuda")
        
        inputs = (x, t, h_t, h_u, c_t, c_u, control_scale, un_scale)

        input_names = ['x', 't', 'h_t', 'h_u','c_t', 'c_u', 'control_scale', 'un_scale']
        ## 输入 图片 提示 timestamp 
        # controls = model(*inputs)

        output_names = ['x_out']
        
        dynamic_table = {
            'x' : {0 : 'bs', 2 : 'H', 3 : 'W'}, 
            'h_t' : {0 : 'bs', 2 : '8H', 3 : '8W'}, 
            'h_u' : {0 : 'bs', 2 : '8H', 3 : '8W'}, 
            # 't_in' : {0 : 'bs'},
            'c_t' : {0 : 'bs'},
            'c_u' : {0 : 'bs'},
            'x_out': {0 : 'bs', 2 : 'H', 3 : 'W'},
        }
        

    elif model_name == 'unet':

        x_in = torch.randn(batch_size, 4, H//8, W//8, dtype=torch.float32).to("cuda")
        t_in = torch.zeros(batch_size, dtype=torch.int64).to("cuda") ## timestep
        c_in = torch.randn(batch_size, 77, 768, dtype=torch.float32).to("cuda") ## text_emb
        ctrl = []

        latent_height = H // 8
        latent_width  = W // 8

        ctrl.append(torch.randn(batch_size, 320, latent_height, latent_width, dtype=torch.float32).to("cuda"))
        ctrl.append(torch.randn(batch_size, 320, latent_height, latent_width, dtype=torch.float32).to("cuda"))
        ctrl.append(torch.randn(batch_size, 320, latent_height, latent_width, dtype=torch.float32).to("cuda"))
        ctrl.append(torch.randn(batch_size, 320, latent_height//2, latent_width//2, dtype=torch.float32).to("cuda"))
        ctrl.append(torch.randn(batch_size, 640, latent_height//2, latent_width//2, dtype=torch.float32).to("cuda"))
        ctrl.append(torch.randn(batch_size, 640, latent_height//2, latent_width//2, dtype=torch.float32).to("cuda"))
        ctrl.append(torch.randn(batch_size, 640, latent_height//4, latent_width//4, dtype=torch.float32).to("cuda"))
        ctrl.append(torch.randn(batch_size, 1280, latent_height//4, latent_width//4, dtype=torch.float32).to("cuda"))
        ctrl.append(torch.randn(batch_size, 1280, latent_height//4, latent_width//4, dtype=torch.float32).to("cuda"))
        ctrl.append(torch.randn(batch_size, 1280, latent_height//8, latent_width//8, dtype=torch.float32).to("cuda"))
        ctrl.append(torch.randn(batch_size, 1280, latent_height//8, latent_width//8, dtype=torch.float32).to("cuda"))
        ctrl.append(torch.randn(batch_size, 1280, latent_height//8, latent_width//8, dtype=torch.float32).to("cuda"))
        ctrl.append(torch.randn(batch_size, 1280, latent_height//8, latent_width//8, dtype=torch.float32).to("cuda"))

        inputs = (x_in, t_in, c_in, *ctrl)
        model = ControlledUnetWrapperModel(model, only_mid_control=False)

        output_names = ['x_out']

        input_names = [
            'x_in',  't_in', 'c_in',
            'ctrl0', 'ctrl1', 'ctrl2', 'ctrl3',
            'ctrl4', 'ctrl5', 'ctrl6', 'ctrl7',
            'ctrl8', 'ctrl9', 'ctrl10', 'ctrl11',
            'ctrl12',
        ]

        dynamic_table = {
            'x_in' : {0 : 'bs', 2 : 'H', 3 : 'W'}, 
            't_in' : {0 : 'bs'},
            'c_in' : {0 : 'bs'},
            'ctrl0': {0 : 'bs'},
            'ctrl1': {0 : 'bs'},
            'ctrl2': {0 : 'bs'},
            'ctrl3': {0 : 'bs'},
            'ctrl4': {0 : 'bs'},
            'ctrl5': {0 : 'bs'},
            'ctrl6': {0 : 'bs'},
            'ctrl7': {0 : 'bs'},
            'ctrl8': {0 : 'bs'},
            'ctrl9': {0 : 'bs'},
            'ctrl10': {0 : 'bs'},
            'ctrl11': {0 : 'bs'},
            'ctrl12': {0 : 'bs'},
            'x_out':  {0 : 'bs'},
        }

    elif model_name == 'vae_decoder':

        x_in = torch.randn(1, 4, H//8, W//8, dtype=torch.float32).to("cuda")

        inputs = (x_in,)

        output_names = ['images']
        
        input_names = [
            'x_in', 
        ]

        dynamic_table = {
            'x_in' : {0 : 'bs', 2 : 'H', 3 : 'W'}, 
            'images':  {0 : 'bs'},
        }
    
    elif model_name == 'vd':
        x_in = torch.randn(1, 4, H//8, W//8, dtype=torch.float32).to("cuda")
        inputs = (x_in,)
        output_names = ['images']
        input_names = [
            'x_in', 
        ]
        dynamic_table = {
            'x_in' : {0 : 'bs', 2 : 'H', 3 : 'W'}, 
            'images':  {0 : 'bs'},
        }

    elif model_name == 'clip':
        input_ids = torch.zeros(batch_size, 77, dtype=torch.int32, device='cuda')
        inputs = (input_ids,)

        output_names = ['text_embeddings']
        
        input_names = ['input_ids']

        dynamic_table = {
            'input_ids' : {0 : 'bs'}, 
            'text_embeddings':  {0 : 'bs'},
            'pooler_output': {0: 'bs'}
        }


    model(*inputs)

    if model_name == 'ddim':
        onnx_export(
            model=model, model_name=model_name,
            inputs=inputs, output_names=output_names, 
            input_names=input_names, dynamic_axes=dynamic_table,
            out_path=out_path, convert_fp16=convert_fp16, const_folding=const_folding,
            # input_shapes = input_shapes,
        )
    else:
        onnx_export(
            model=model, model_name=model_name,
            inputs=inputs, output_names=output_names, 
            input_names=input_names, dynamic_axes=dynamic_table,
            out_path=out_path, convert_fp16=convert_fp16, const_folding=const_folding,
        )


def compile(input_path, out_path, model_name, batch_size=1, opts = ['--fp16']):

    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # out_path = f'{out_dir}/{model_name}.engine'

    latent_height = H // 8
    latent_width  = W // 8
    latent_channel  = 4
    img_hegint = H
    img_width = W
    img_channel = 3
    embedding_dim = 768
    text_maxlen = 77


    if model_name == 'control':

        input_shapes = {
            'x_in' : [batch_size, latent_channel, latent_height, latent_width],
            'h_in' : [batch_size, img_channel, img_hegint, img_width], 
            't_in' : [batch_size], 
            'c_in' : [batch_size, text_maxlen, embedding_dim],
        }
        
    elif model_name == 'unet_with_spec_control':

        input_shapes = {
            'x_in' : [batch_size, latent_channel, latent_height, latent_width],
            'h_in' : [batch_size, img_channel, img_hegint, img_width], 
            't_in' : [1], 
            'c_in' : [batch_size, text_maxlen, embedding_dim],
            'control_scale': [13],
        }

    elif model_name == 'uc1b':
        # x, t, h_t, h_u, c_t, c_u, control_scale, un_scale
        input_shapes = {
            'x' : [batch_size, latent_channel, latent_height, latent_width],
            't' : [1], 
            'h_t' : [batch_size, img_channel, img_hegint, img_width], 
            'h_u' : [batch_size, img_channel, img_hegint, img_width],
            'c_t' : [batch_size, text_maxlen, embedding_dim],
            'c_u' : [batch_size, text_maxlen, embedding_dim],
            'control_scale': [13],
            'un_scale': [1],
        }


    elif model_name == 'vae_decoder':
        # x, t, h_t, h_u, c_t, c_u, control_scale, un_scale
        input_shapes = {
            'x_in' : [batch_size, latent_channel, latent_height, latent_width],
        }

    elif model_name == 'vd':
        # x, t, h_t, h_u, c_t, c_u, control_scale, un_scale
        input_shapes = {
            'x_in' : [batch_size, latent_channel, latent_height, latent_width],
        }

    elif model_name == 'clip':
        input_shapes = {
            'input_ids' : [batch_size, 77],
        }

    # elif model_name == 'ddim':
    #     input_shapes = {
    #         'x' : [batch_size, latent_channel, latent_height, latent_width], 
    #         't' : [1], 
    #         'c_t' : [batch_size, text_maxlen, embedding_dim],
    #         'c_u' : [batch_size, text_maxlen, embedding_dim], 
    #         'h_t' : [batch_size, img_channel, img_hegint, img_width], 
    #         'h_u' : [batch_size, img_channel, img_hegint, img_width], 
    #         'un_scale': [],
    #     }
    
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
    
    shape_str =  ','.join([k + ':' + ('x'.join([str(vv) for vv in v]) if len(v) > 0 else '0') for k, v in input_shapes.items()])
    
    cmd = f"trtexec --onnx={input_path} --saveEngine={out_path} --optShapes={shape_str}"
    
    if opts:
        cmd +=' ' + ' '.join(opts)
        
    print(f' ============ compile {model_name} begine =========')
    print(f' cmd ==> {cmd}')
    t0 = time.time()
    os.system(cmd)
    dt = time.time() - t0
    print(f' ============ compile {model_name} end, dt={dt}, output={out_path} =========')


if __name__ == '__main__':
    
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(f'/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
    model = model.cuda()
    
    # ### ------------ 编译1batch ---------------------- ##

    control_model = model.control_model
    unet_model = model.model.diffusion_model
    
    vd_model = VD(decoder=model.first_stage_model.decoder, post_quant_conv=model.first_stage_model.post_quant_conv)
    vd_onnx = './onnx/vd/model.onnx'
    vd_plan = './engine/vd.engine'


    export(model=vd_model, model_name='vd', out_path=vd_onnx, const_folding=True, convert_fp16=False, batch_size=1)
    compile(input_path=vd_onnx, out_path=vd_plan, model_name='vd', batch_size=1)

    # 联合
    uc1b_model = UC1B(unet_model=unet_model, control_model = control_model, only_mid_control=False)
    uc1b_onnx = './onnx/uc1b/model.onnx'
    uc1b_plan = './engine/uc1b.engine'

    export(model=uc1b_model, model_name='uc1b', out_path=uc1b_onnx, const_folding=True, convert_fp16=False, batch_size=1)
    # compile(input_path=uc1b_onnx, out_path=uc1b_plan, model_name='uc1b', batch_size=1)


    # # 编译失败，需要量化
    # uc1b_fp8_plan = './engine/uc1b_fp8.engine'
    # compile(input_path=uc1b_onnx, out_path=uc1b_fp8_plan, model_name='uc1b', batch_size=1, opts=['--fp8'])



    uc1b_opt5_plan = './engine/uc1b_opt5.engine'
    compile(input_path=uc1b_onnx, out_path=uc1b_opt5_plan, model_name='uc1b', batch_size=1, opts=['--fp16', '--builderOptimizationLevel=5'])







    # ## #显存不够
    # # ddim_model = DDIM_(unet_model=unet_model, control_model=control_model, only_mid_control=False)
    # # ddim_onnx = './onnx/ddim/model.onnx'
    # # ddim_plan = './engine/ddim.engine'
    
    # # export(model=ddim_model, model_name='ddim', 
    # #     out_dir='./onnx/ddim', const_folding=True, convert_fp16=False, batch_size=1)
    # # compile(input_path=ddim_onnx, out_dir='./engine/', model_name='ddim', batch_size=1)
    
    
    ##  ----------------  编译2batch ----------------------------
    

    # control_model = model.control_model
    # control_onnx = './onnx2/control/model.onnx'
    # control_plan = './engine2/control.engine'
    
    
    # unet_model = model.model.diffusion_model
    # unet_onnx = './onnx2/unet/model.onnx'
    # unet_plan = './engine2/unet.engine'
    
    # export(model=control_model, model_name='control', out_dir='./onnx2/control', const_folding=True, convert_fp16=False, batch_size=2)
    # compile(input_path=control_onnx, out_dir='./engine2/', model_name='control', batch_size=2)
    
    # export(model=unet_model, model_name='unet', out_dir='./onnx2/unet', const_folding=True, convert_fp16=False, batch_size=2)
    # compile(input_path=unet_onnx, out_dir='./engine2/', model_name='unet', batch_size=2)



    # ## 联合
    # unet_with_spec_control_model = UnetWithSpecControl(unet_model=unet_model, control_model = control_model, only_mid_control=False)
    # unet_with_spec_control_onnx = './onnx2/unet_with_spec_control/model.onnx'
    # unet_with_spec_control_plan = './engine2/unet_with_spec_control.engine'
    
    # os.makedirs('./onnx2/unet_with_spec_control/', exist_ok=True)
    # os.makedirs('./engine2/', exist_ok=True)
    # export(model=unet_with_spec_control_model, model_name='unet_with_spec_control', 
    #        out_dir='./onnx2/unet_with_spec_control', const_folding=True, convert_fp16=False, batch_size=2)
    # compile(input_path=unet_with_spec_control_onnx, out_dir='./engine2/', model_name='unet_with_spec_control', batch_size=2)
    


   