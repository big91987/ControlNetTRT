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
from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel


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
    

class VD(torch.nn.Module):
    def __init__(self, decoder, post_quant_conv):
        super().__init__()
        self.decoder = decoder
        self.post_quant_conv = post_quant_conv

    def forward(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec


## 图片大小
H = 256
W = 384

def export(model, model_name, out_path, convert_fp16=True, const_folding=True, batch_size = 1):

    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    input_shapes = {}
    
    if model_name == 'uc1b':
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


    model(*inputs)

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

    latent_height = H // 8
    latent_width  = W // 8
    latent_channel  = 4
    img_hegint = H
    img_width = W
    img_channel = 3
    embedding_dim = 768
    text_maxlen = 77

    if model_name == 'uc1b':
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

    elif model_name == 'vd':
        # x, t, h_t, h_u, c_t, c_u, control_scale, un_scale
        input_shapes = {
            'x_in' : [batch_size, latent_channel, latent_height, latent_width],
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
    compile(input_path=uc1b_onnx, out_path=uc1b_plan, model_name='uc1b', batch_size=1)


    clip_onnx = './onnx/clip/model.onnx'
    clip_plan = './engine/clip.engine'
    
    clip_model =  CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
    os.makedirs(os.path.dirname(clip_onnx), exist_ok=True)
    os.makedirs(os.path.dirname(clip_plan), exist_ok=True)
    input_ids = torch.ones(1, 77, dtype=torch.int64).to('cuda')
    torch.onnx.export(
        clip_model,
        (input_ids),
        clip_onnx,
        export_params=True,
        opset_version = 18,
        do_constant_folding = True,
        keep_initializers_as_inputs = True,
        input_names=['input_ids'],
        output_names=['context', 'pooled_output'],
        dynamic_axes={
            'input_ids': {0:'bs'},
            'context': {0:'bs'},
            'pooled_output': {0:'bs'},
        }
    )
    os.system(f'trtexec --onnx={clip_onnx} --saveEngine={clip_plan}')