import os
import torch
from tqdm import tqdm
from diffusers import DiffusionPipeline
import numpy as np
from PIL import Image
from typing import Union


def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image

class DDIMInversion:
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(self.model.device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )  # dict keys: input_ids, attention_mask
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in tqdm(range(self.num_ddim_steps)):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def invert(self, image, prompt, offsets=(0, 0, 0, 0)):
        print("DDIM inversion...")
        self.init_prompt(prompt)
        image_gt = load_512(image, *offsets)
        latent = self.image2latent(image_gt)
        ddim_latents = self.ddim_loop(latent)
        image_rec = self.latent2image(latent)

        return ddim_latents[-1]

    def __init__(self, model, num_ddim_steps=50):
        self.model = model
        self.num_ddim_steps = num_ddim_steps
        self.model.scheduler.set_timesteps(self.num_ddim_steps)



def diffusion_step(model, latents, context, t, guidance_scale):
    latents_input = torch.cat([latents] * 2)  # latents_input 2, 4, 64,64
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)[
        "sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents


@torch.no_grad()
def latent2image(vae, latents, return_type='np'):
    latents = 1 / 0.18215 * latents.detach()
    image = vae.decode(latents)['sample']
    if return_type == 'np':
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)

    return image

@torch.no_grad()
def DDIM(
        model,
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        latent=None):
    print("DDIM Sampling...")

    # text_embeddings
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]

    # uncond embeddings
    uncond_input = model.tokenizer( [""] , padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    # latent
    if latent is None:
        latent = torch.randn((1, 4, 64, 64))
    latents = latent.expand(1,4, 64, 64).to(model.device)

    # model scheduler
    model.scheduler.set_timesteps(num_inference_steps)

    for i, t in enumerate(tqdm(model.scheduler.timesteps[-num_inference_steps:])):
        context = torch.cat([uncond_embeddings, text_embeddings])
        latents = diffusion_step(model, latents, context, t, guidance_scale)

    image = latent2image(model.vae, latents)
    return image





def save_images(images, num_rows=1, offset_ratio=0.02,img_path='./output/test.jpg'):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    pil_img.save(img_path)


def AdaIN(content_feat, style_feat,eps=1e-5):
    # compute mean and std
    c_mean, c_std = torch.mean(content_feat, dim=(2, 3), keepdim=True), torch.std(content_feat, dim=(2, 3), keepdim=True)
    s_mean, s_std = torch.mean(style_feat, dim=(2, 3), keepdim=True), torch.std(style_feat, dim=(2, 3), keepdim=True)
    # norm
    norm_content_feat = (content_feat - c_mean) / (c_std + eps)
    stylized_feat = norm_content_feat * s_std + s_mean

    return stylized_feat

def load_sd(pretrained_model_name_or_path):
    ldm_stable = DiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path #"CompVis/stable-diffusion-v1-4"
    ).to("cuda")
    return ldm_stable

def load_my_sd(pretrained_model_name_or_path,cd_output,sty_init,obj_init,time):
    ldm_stable = DiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path #"CompVis/stable-diffusion-v1-4"
    ).to("cuda")
    
    cd_output=cd_output+'/'+sty_init+'-'+obj_init+'-'+time+'/'
    print("loading trained checkpoints")
    ldm_stable.unet.load_attn_procs(
        cd_output, weight_name="pytorch_custom_diffusion_weights.bin"
    )
    ldm_stable.load_textual_inversion(cd_output, weight_name="<new1>.bin")
    ldm_stable.to('cuda')

    return ldm_stable, "Finish Load Model!"


def run_inference_inversion(ldm_stable, image, sty_init, obj_init, steps=50):
    # ddim inversion
    inv_prompt = 'A ' + obj_init + ' in the style of ' + sty_init
    ddim_inversion = DDIMInversion(ldm_stable, steps)
    x_t = ddim_inversion.invert(image, inv_prompt, offsets=(0, 0, 0, 0))
    return x_t

def run_inference_sampling(ldm_stable, x_t, sty_init, obj_tar, outdir_current='./inf_output/', scale=2.5, steps=50,rand_scale=0):
    os.makedirs(outdir_current, exist_ok=True)
    # latent
    mask_orig = torch.rand(64, 64)
    mask = mask_orig.ge(rand_scale).to('cuda')
    x_t_rand = torch.randn((1, 4, 512 // 8, 512 // 8)).to('cuda')
    output_feat = AdaIN(x_t_rand, x_t)
    x_t_new = mask * x_t + (~mask) * output_feat

    # ddim sampling
    sample_prompt = 'A ' + obj_tar + ' in the style of ' + sty_init
    image_ori = DDIM(ldm_stable, [sample_prompt], num_inference_steps=steps, guidance_scale=scale, latent=x_t_new)

    # save images
    save_images([image_ori[0]], img_path=outdir_current + obj_tar + '.jpg')

    return image_ori[0], sample_prompt


def run_inference(ldm_stable, image, sty_init, obj_init, obj_tar, outdir_current='./inf_output/', scale=2.5,steps=50,rand_scale=0 ):
    os.makedirs(outdir_current,  exist_ok = True)

    # ddim inversion
    inv_prompt='A '+obj_init+' in the style of '+sty_init
    ddim_inversion = DDIMInversion(ldm_stable,steps)
    x_t = ddim_inversion.invert(image, inv_prompt, offsets=(0, 0, 0, 0))

    # latent
    mask_orig = torch.rand(64, 64)
    mask = mask_orig.ge(rand_scale).to('cuda')
    x_t_rand = torch.randn((1, 4, 512 // 8, 512 // 8)).to('cuda')
    output_feat = AdaIN(x_t_rand, x_t)
    x_t_new = mask * x_t + (~mask) * output_feat

    # ddim sampling
    sample_prompt='A ' + obj_tar + ' in the style of ' + sty_init
    image_ori = DDIM(ldm_stable, [sample_prompt],num_inference_steps=steps,guidance_scale=scale, latent=x_t_new)

    # save images
    save_images([image_ori[0]], img_path=outdir_current + obj_tar+'.jpg' )
    
    return image_ori[0],sample_prompt
