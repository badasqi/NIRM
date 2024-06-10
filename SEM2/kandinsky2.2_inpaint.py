import torch

cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

import warnings
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
import numpy as np

# Suppress specific warning
warnings.filterwarnings("ignore", message="`resume_download` is deprecated and will be removed in version 1.0.0.")

device = torch.device('cuda')

# Load the inpainting pipeline
pipe = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint",
    torch_dtype=torch.float32  # Use float32 for CPU
)
pipe.enable_model_cpu_offload()

# Define the prompt and load the initial image
prompt = "a hat"
init_image = load_image(
    "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png"
)

# Create a mask to define the area for inpainting
mask = np.zeros((768, 768), dtype=np.float32)
mask[:250, 250:-250] = 1

# Perform inpainting
out = pipe(
    prompt=prompt,
    image=init_image,
    mask_image=mask,
    height=768,
    width=768,
    num_inference_steps=150,
)

# Save the output image
image = out.images[0]
image.save("cat_with_hat.png")

print("Inpainting completed and image saved as 'cat_with_hat.png'.")
