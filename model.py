from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

model = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

input_image = Image.open(r"C:\Users\arulk\OneDrive\Desktop\Diffusion Model\sample.jpg")


prompt = "Make the image look like a painting"  

output_image = model(prompt=prompt, image=input_image)


output_image.save("output_image.png")

