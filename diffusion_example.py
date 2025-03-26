import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from torch import nn

# Załaduj klasę, która powoduje błąd
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

# Dodaj do bezpiecznych globalnych klas
torch.serialization.add_safe_globals([ModelCheckpoint])

# Ścieżka do pliku .ckpt
model_path = "./sd-v1-4-full-ema.ckpt"  # Zmień na ścieżkę do swojego pliku .ckpt


# Funkcja do załadowania modelu .ckpt
def load_ckpt_model(ckpt_path):
    # Załaduj model z pliku .ckpt
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Załaduj odpowiednie komponenty modelu
    model = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', local_files_only=True)

    # Załaduj wagę do odpowiednich komponentów (model i konfiguracja)
    model.unet.load_state_dict(checkpoint['state_dict'], strict=False)
    model.to("mps")  # Jeśli masz GPU, przenieś model na GPU
    return model


# Załaduj model z pliku .ckpt
pipe = load_ckpt_model(model_path)

# Generowanie obrazu z przykładowym promptem
prompt = "Lord Vader is freediving. That means he's in the rashsuit with long freediving fins"
image = pipe(prompt).images[0]

# Zapis wygenerowanego obrazu
image.save("output.png")

print("Obraz zapisany jako output.png")