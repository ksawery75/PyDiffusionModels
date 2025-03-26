from diffusers import StableDiffusionPipeline
import torch

# Załaduj model Stable Diffusion
# model_id = "CompVis/stable-diffusion-v-1-4-original"
model_id = "sd-v1-4-full-ema.ckpt"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Tekstowe zapytanie, które może pochodzić z przetworzonych danych sensorycznych
prompt = "Military vehicles on a battlefield with radar images visible and soldiers in the terrain"

# Generowanie obrazu
image = pipe(prompt).images[0]
image.show()
