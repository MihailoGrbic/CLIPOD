import random

import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

INIT_THRESHOLD = 1e-3
MIN_BBOX_DIM = 10
random.seed(3141592)
np.random.seed(3141592)

class CLIPObjectDetector:
    def __init__(self, model_name, device="cpu", batch_size=8):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = device
        self.batch_size = batch_size

        self.model.to(device).eval()

    def _get_probs(self, texts, images):
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        # Move inputs to GPU
        for key, val in inputs.items():
            inputs[key] = val.to(self.device)

        outputs = self.model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        probs = probs.cpu().numpy()

        return probs

    def _generate_random_bboxes(self, img_width, img_height, num):
        lefts = np.random.uniform(0, img_width - MIN_BBOX_DIM, num).astype(int)
        tops = np.random.uniform(0, img_height - MIN_BBOX_DIM, num).astype(int)
        widths = np.zeros(num)
        heights = np.zeros(num)
        for i, (left, top) in enumerate(zip(lefts, tops)):
            widths[i] = np.random.uniform(MIN_BBOX_DIM, img_width - left)
            heights[i] = np.random.uniform(MIN_BBOX_DIM, img_height - top)
        rights = lefts + widths
        downs = tops + heights

        return np.stack([lefts, tops, rights, downs], axis=-1).astype(int)

    def _crop_img(self, image, bbox):
        left, top, right, down = bbox[0], bbox[1], bbox[2], bbox[3]

        if self.crop:
            proc_image = image[top:down, left:right, :]
        else:
            proc_image = np.copy(image)
            proc_image[:top, :, :] = 0
            proc_image[top:down, :left, :] = 0
            proc_image[top:down, right:, :] = 0
            proc_image[down:, :, :] = 0

        return proc_image

    def _random_detect(self, image, text_prompt, num_tries):
        img_height, img_width, _ = image.shape
        bboxes = self._generate_random_bboxes(img_width, img_height, num_tries)

        confidence_scores = np.zeros(num_tries)

        # Iterate through bboxes in batches
        for i in range(0, num_tries, self.batch_size):
            bbox_batch = bboxes[i:i+self.batch_size]
            image_batch = []
            for bbox in bbox_batch:
                proc_image = self._crop_img(image, bbox)
                image_batch.append(proc_image)

            # Run images through the model
            probs = self._get_probs(text_prompt, image_batch)
            #probs = np.zeros((len(bbox_batch), 2))
            confidence_scores[i:i+self.batch_size] = probs[:, 0]

        detections = []
        for i, score in enumerate(confidence_scores):
            if score > 0.5:
                detections.append({'bbox': bboxes[i].tolist(), 'score':score})
        
        return detections

    def detect(self, image, cat_names, text_strat="one-negative", bbox_strat="random", crop=True,
            negative_text="background", num_other_cats=10, num_tries=100):

        self.crop = crop

        # Initial class filtering
        init_probs = self._get_probs(cat_names, image)

        promising_cats = []
        negative_cats = []
        for i, cat_name in enumerate(cat_names):
            if init_probs[0, i] > INIT_THRESHOLD:
                promising_cats.append(cat_name)
            else:
                negative_cats.append(cat_name)


        all_detections = []
        # Loop through all categories that may be present on the image
        for target_cat in tqdm(promising_cats):

            # Generate text prompt
            if text_strat == "one-negative":
                text_prompt = [target_cat, negative_text]
            elif text_strat == "negative-cats":
                text_prompt = [target_cat]
                text_prompt.extend(random.sample(negative_cats, min(num_other_cats, len(negative_cats))))
            elif text_strat == "promising-cats":
                text_prompt = [target_cat]
                text_prompt.extend(random.sample(promising_cats, min(num_other_cats, len(promising_cats))))
            else:
                raise ValueError(
                    f"Invalid text prompting strategy: {text_strat}. "
                    "Must be one of [one-negative, negative-cats, promising-cats]")
            
            # Find bboxes
            if bbox_strat == "random":
                detections = self._random_detect(image, text_prompt, num_tries)
            elif bbox_strat == "random+search":
                pass
            elif bbox_strat == "split+merge":
                pass
            
            # Add field for category name 
            for detection in detections: detection['cat'] = target_cat

            #TODO Non-maximum suppression
            pass

            all_detections.extend(detections)

            

        return all_detections
