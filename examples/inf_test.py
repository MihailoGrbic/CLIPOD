from PIL import Image

from transformers import CLIPProcessor, CLIPModel

model_name = "openai/clip-vit-base-patch32"
# model_name = "openai/clip-vit-large-patch14"
#model_name = "openai/clip-vit-large-patch14-336"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

#image = Image.open("img/dog-cat-bird.jpg")
image = Image.open("img/not_a_car.jpg")
# image = Image.open("img/cat.jpg")
# image = Image.open("img/dog.jpg")
# image = Image.open("img/bird.jpg")

# inputs = processor(text=["a photo of a cat", "a photo of a dog", "a photo of a bird"],
#                    images=image, return_tensors="pt", padding=True)
inputs = processor(text=["a photo of a bicycle", "a photo of a car"],
                   images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)

logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

print(probs)
