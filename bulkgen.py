import torch
from diffusers import AutoPipelineForText2Image
from diffusers import StableDiffusionPipeline
from datetime import datetime
import logging
import os
import numpy as np

logging.basicConfig(
    filename="bulkgen.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

model_path = "models/" # Local model path
if not os.path.exists(model_path):
    logging.error(f"Model directory does not exist: {model_path}")
    raise RuntimeError(f"Model directory does not exist: {model_path}")

try:
    logging.info(f"Loading local model from {model_path}")
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        use_fast_tokenizer=True,
        use_auth_token=False,
        local_files_only=True,
        safety_checker=None,
    )
    pipe.to(device)

except Exception as e:
    logging.error(f"Model loading failed: {str(e)}")
    raise RuntimeError(f"Failed to load model: {str(e)}")

for num in range(30) : # Choose how many images to generate
    """
    Generate an image based on the given prompt and negative prompt.
    
    Args:
        prompt (str): The text prompt for image generation
        negative_prompt (str): The negative prompt to avoid certain elements
        
    Returns:
        PIL.Image: Generated image
    """
    try:
        logging.info(f"Generating image for prompt...")
        seed = torch.randint(0, 2**32, (1,)).item()
        logging.info(f"Using seed: {seed}")
        generator = torch.Generator(device=device).manual_seed(seed)

        image = pipe(
            prompt="prompt", # Write prompt 
            negative_prompt="lowres, (bad quality, worst quality:1.2), bad anatomy, deformed, missing digits, extra digits, broken hands, sketch, jpeg artifacts, ugly, poorly drawn, blurry eyes, watermark, simple background, transparent background,",
            num_inference_steps=30,
            height=800,
            width=440,
            guidance_scale=7.0,
            generator=generator,
            output_type="np"
        ).images[0]

        from PIL import Image
        image = Image.fromarray(np.clip(image * 255, 0, 255).astype(np.uint8))
        save_dir = "bulk_generated_images"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{save_dir}/{timestamp}_{num}.png"
        image.save(image_filename)
        logging.info(f"Image saved as {image_filename}")
        
    except Exception as e:
        logging.error(f"Image generation failed: {str(e)}")
        raise RuntimeError(f"Image generation failed: {str(e)}")