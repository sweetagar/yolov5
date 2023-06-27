from PIL import Image

def get_image_mode(image_path):
    with Image.open(image_path) as image:
        mode = image.mode
    return mode

image_path = "./baby.jpg"
image_mode = get_image_mode(image_path)
print("Image mode:", image_mode)
