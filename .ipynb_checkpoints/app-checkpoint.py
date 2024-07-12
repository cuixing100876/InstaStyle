from PIL import Image as PILImage
from pathlib import Path
import gradio as gr
from accelerate import Accelerator
import cv2
import numpy as np

from train import run_train
from inference import run_inference ,load_sd,load_my_sd,run_inference_inversion,run_inference_sampling

import datetime

def get_time():
    time=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return time

# initial image generation
def ref_function(pretrained_model_name_or_path,image, sty_init, obj_init,obj_tars,init_scale,init_steps,init_rand_scale):
    print("reference function")
    time=get_time()
    ldm_stable=load_sd(pretrained_model_name_or_path)
    
    ref_img_list=[]
    ref_img_cap_list=[]
    if len(obj_tars) != 0:
        obj_tar_list=obj_tars.split(",")
    else:
        obj_tar_list=[]
    x_t = run_inference_inversion(ldm_stable, image, sty_init, obj_init, steps=init_steps)
    for obj_tar in obj_tar_list:
        out_img,sample_prompt=run_inference_sampling(ldm_stable, x_t, sty_init, obj_tar,  outdir_current='./ref_output_tmp'+time+'/', scale=init_scale,steps=init_steps,rand_scale=init_rand_scale)
        # out_img,sample_prompt=run_inference(ldm_stable, image, sty_init, obj_init, obj_tar,  outdir_current='./ref_output_tmp'+time+'/', scale=init_scale,steps=init_steps,rand_scale=init_rand_scale )
        ref_img_list.append(out_img)
        ref_img_cap_list.append((out_img,sample_prompt))
    return ref_img_list,obj_tar_list,ref_img_cap_list,time,"Finish Initial Stylized Image Generation!"

def add_id_to_image(image, id_int,obj):
    id_text=str(id_int)+":"+obj
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 2

    height, width, channels = image.shape

    # text position
    text_size = cv2.getTextSize(id_text, font, font_scale, font_thickness)[0]
    text_x = (width + 2*text_size[1] + 20 - text_size[0]) // 2
    text_y = height + 2*text_size[1] + 10

    # new image
    new_image = np.ones((height + 2*text_size[1] + 20, width + 2*text_size[1] + 20, channels), dtype=np.uint8)*255

    # paste old image to new image
    paste_x = (new_image.shape[1] - width) // 2
    paste_y = text_size[1]
    new_image[paste_y:paste_y + height, paste_x:paste_x + width] = image

    # past text to new image
    cv2.putText(new_image, id_text, (text_x , text_y), font, font_scale, (0, 0, 0), font_thickness)

    return new_image

# image selection
def select_image(selected_ids, ref_img_cap_list,image,obj_init,sty_init,time):
    instance_data_dir='./ref_output_selected'+time+'/'
    sample_prompt='A ' + obj_init + ' in the style of ' + sty_init
    selected_imgs=[image]
    selected_imgs_cap=[(image,sample_prompt)]

    if len(selected_ids)!=0:
        selected_ids_list=selected_ids.split(',')
    else:
        selected_ids_list=[]

    for id in selected_ids_list:
        id = int(id.strip())
        if id >= 0 and id <= len(ref_img_cap_list)-1:
            selected_imgs_cap.append(ref_img_cap_list[id])  
            selected_imgs.append(ref_img_cap_list[id][0])
    save_tmp(selected_imgs_cap,instance_data_dir)
    return selected_imgs,instance_data_dir


def save_tmp(selected_imgs_cap,instance_data_dir):
    tmp_dir = Path(instance_data_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print("Number of selected images:", len(selected_imgs_cap))
    
    for img,img_name in selected_imgs_cap:
        pil_img = PILImage.fromarray(img)
        img_path = tmp_dir / f"{img_name}.png"
        pil_img.save(img_path)
    return tmp_dir


accelerator = Accelerator()
def train(pretrained_model_name_or_path, freeze_model, enable_xformers_memory_efficient_attention, instance_data_dir, sty_init, obj_init,time,dataloader_num_workers,
                learning_rate, lr_warmup_steps, lr_scheduler_constant ,adam_beta1,adam_beta2,adam_weight_decay,adam_epsilon,max_grad_norm,
                max_train_steps, train_batch_size,gradient_accumulation_steps,checkpointing_steps,  train_output_root):
    with accelerator.main_process_first():
        run_train(pretrained_model_name_or_path, freeze_model, enable_xformers_memory_efficient_attention, instance_data_dir, sty_init, obj_init,time,dataloader_num_workers,
                learning_rate, lr_warmup_steps, lr_scheduler_constant ,adam_beta1,adam_beta2,adam_weight_decay,adam_epsilon,max_grad_norm,
                max_train_steps, train_batch_size,gradient_accumulation_steps,checkpointing_steps,  train_output_root)
    return  "Finish Prompt Refinement!"


def inference(my_ldm_stable, image, sty_init, obj_init, obj_tar,scale,steps,rand_scale,time):
    print("inferencing")
    x_t = run_inference_inversion(my_ldm_stable, image, sty_init, obj_init, steps=steps)
    out_img, sample_prompt = run_inference_sampling(my_ldm_stable, x_t, sty_init,obj_tar, outdir_current='./ref_output_final'+time+'/', scale=scale,steps=steps,rand_scale=rand_scale )

    # out_img,sample_prompt=run_inference(my_ldm_stable, image, sty_init, obj_init,obj_tar, outdir_current='./ref_output_final'+time+'/', scale=scale,steps=steps,rand_scale=rand_scale )
    return out_img


demo = gr.Blocks()

with demo:
    # =====================# User Input=====================
    gr.Markdown("# User Input")
    with gr.Row():
        image = gr.Image(label="Image", scale=0.7,value='./style-images/haunted house/1 (2).jpg')
        with gr.Column():
            sty_init = gr.Textbox(label="Style",value='dark',placeholder="a single word, e.g., painting, watercolor, dark")
            obj_init = gr.Textbox(label="Object",value='house')
            pretrained_model_name_or_path = gr.Textbox(label="Pretrained Model Name or Path",value="/home/vcis8/Userlist/cuixing3/sd-ckpt/sd")
            obj_tars = gr.Textbox(label="Original generation objects",
                                  value="cat, lighthouse, goldfish,table lamp, tram, tower, cup, desk, chair, pot")
            # cat, lighthouse, volcano, goldfish,table lamp, tram, palace, tower, cup, desk, chair, pot, laptop, door,  car

            with gr.Accordion(label="Initial Stylized Image Generation Options",open=False):
                # instance_data_dir = gr.Textbox(label="Generated instance data dir", value='ref_output_selected')
                init_scale = gr.Number(label="Guidance Scale", value=2.5, step=0.1,minimum=0, maximum=10 )
                init_steps = gr.Number(label="Steps", value=50, step=1,minimum=1,maximum=1000)
                init_rand_scale = gr.Number(label="Rand Scale", value=0, step=0.1,minimum=0,maximum=1)

    # =====================# Initial Stylized Image Generation=====================
    gr.Markdown("# Initial Stylized Image Generation")
    # generation
    ref_img_list = gr.State([])
    ref_img_cap_list = gr.State([])
    obj_tar_list=gr.State([])
    time= gr.State([])

    with gr.Row():
        submit_btn = gr.Button("Initial Stylized Image Generation")
        submit_output_flag = gr.Textbox(label="Initial Stylized Image Generation Progress Bar")
    with gr.Row():
        ref_gallery = gr.Gallery(label="Reference Images", columns=10, rows=2,height=160)
    

    submit_btn.click(ref_function, inputs=[pretrained_model_name_or_path,image, sty_init, obj_init,obj_tars,init_scale,init_steps,init_rand_scale], outputs=[ref_img_list,obj_tar_list,ref_img_cap_list,time,submit_output_flag])
    ref_img_list.change(lambda imgs,obj_tars: [add_id_to_image(img,it,obj) for it,(img,obj) in enumerate(zip(imgs,obj_tars))], inputs=[ref_img_list,obj_tar_list], outputs=ref_gallery)
    
    # selection
    selected_imgs = gr.State([])
    instance_data_dir = gr.State([])
    with gr.Row():
        with gr.Column():
            select_ids = gr.Textbox(label="Enter IDs (English comma separated)", placeholder="e.g., 0, 1, 3, 5")
            finish_select_btn = gr.Button("Human Selection")
        ref_gallery_select = gr.Gallery(label="Reference Images", columns=5, rows=2,height=160)

    finish_select_btn.click(select_image, inputs=[select_ids, ref_img_cap_list, image,obj_init,sty_init, time], outputs=[selected_imgs,instance_data_dir])
    selected_imgs.change(lambda imgs: [img for img in imgs], inputs=selected_imgs, outputs=ref_gallery_select)

    # =====================# Prompt Refinement=====================
    gr.Markdown("# Prompt Refinement")
    with gr.Row():

            with gr.Column():
                with gr.Accordion(label="Training Options", open=False):
                    freeze_model =gr.Dropdown(choices=['crossattn_kv', 'crossattn'],label="Freeze model. ", value="crossattn_kv")
                    enable_xformers_memory_efficient_attention = gr.Dropdown(choices=[True,False],label="enable_xformers_memory_efficient_attention", value=True)
                    dataloader_num_workers=gr.Number(label="Number workers", value=2, step=1,minimum=0,maximum=10)
                    train_output_root = gr.Textbox(label="Checkpoint output", value="./train_output/")

                    max_train_steps = gr.Number(label="Max train steps", value=500, step=1, minimum=100, maximum=2000)
                    train_batch_size = gr.Number(label="Train batch size", value=2, step=1)
                    gradient_accumulation_steps = gr.Number(label="Gradient accumulation steps", value=1, step=1)
                    checkpointing_steps = gr.Number(label="Save checkpointing interval", value=250, step=1)


            with gr.Column():
                with gr.Accordion(label="Learning rate options", open=False):
                    learning_rate = gr.Number(label="Learning rate", value=1e-5, step=1e-5)
                    lr_warmup_steps = gr.Number(label="LR warmup steps", value=0, step=1,minimum=0,maximum=2000)
                    lr_scheduler_constant = gr.Dropdown(choices=["linear", "cosine", "cosine_with_restarts", "polynomial","constant", "constant_with_warmup"],
                                                        label="Lr scheduler constant", value="constant")

                    adam_beta1 = gr.Number(label="Adam beta1", value=0.9)
                    adam_beta2 = gr.Number(label="Adam beta2", value=0.999)
                    adam_weight_decay = gr.Number(label="Adam weight decay", value=1e-2)
                    adam_epsilon = gr.Number(label="Adam epsilon", value=1e-08)
                    max_grad_norm = gr.Number(label="Max grad norm", value=1.0)

    with gr.Row():
        train_btn = gr.Button("Prompt Refinement")
        train_output_flag = gr.Textbox(label="Prompt Refinement Progress Bar")


    train_btn.click(
        fn=train, 
        inputs=[pretrained_model_name_or_path, freeze_model, enable_xformers_memory_efficient_attention, instance_data_dir, sty_init, obj_init,time,dataloader_num_workers,
                learning_rate, lr_warmup_steps, lr_scheduler_constant ,adam_beta1,adam_beta2,adam_weight_decay,adam_epsilon,max_grad_norm,
                max_train_steps, train_batch_size,gradient_accumulation_steps,checkpointing_steps,  train_output_root],
        outputs=[train_output_flag]
    )

    # =====================# Inference=====================
    gr.Markdown("# Inference")
    my_ldm_stable=gr.State()
    with gr.Row():
        load_my_btn = gr.Button("Loading Inference Model")
        load_model_flag = gr.Textbox(label="Loading Progress Bar")
    with gr.Row():
        with gr.Column():
            obj_tar = gr.Textbox(label="Target Object") 
            with gr.Accordion(label="Inference Options",open=True):
                scale = gr.Number(label="Guidance Scale", value=2.5, step=0.1,minimum=0, maximum=10)
                steps = gr.Number(label="Steps", value=50, step=1,minimum=1,maximum=1000)
                rand_scale = gr.Number(label="Rand Scale", value=0.1, step=0.1,minimum=0,maximum=1)

            ref_btn = gr.Button("Inference")
            
        inference_img = gr.Image(label="Inference Output",scale=0.7)
        
    load_my_btn.click(load_my_sd,inputs=[pretrained_model_name_or_path,train_output_root,sty_init,obj_init,time],outputs=[my_ldm_stable,load_model_flag])
    ref_btn.click(inference, inputs=[my_ldm_stable, image, sty_init, obj_init, obj_tar, scale,steps,rand_scale,time], outputs=inference_img)


if __name__ == "__main__":
    demo.launch()
