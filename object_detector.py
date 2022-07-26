import random

import numpy as np
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

from object_detector_config import CLIPObjectDetectorConfig

MIN_BBOX_DIM = 10
random.seed(3141592)
np.random.seed(3141592)

class CLIPObjectDetector:
    def __init__(self, model_name, config={}):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        if isinstance(config, str):
            self.config = CLIPObjectDetectorConfig.from_file(config_path)
        elif isinstance(config, dict):
            self.config = CLIPObjectDetectorConfig(config)
        elif isinstance(config, CLIPObjectDetectorConfig):
            self.config = config
        else:
            raise ValueError(f"Unknown object type for config")

        self.model.to(self.config.device).eval()

    def _get_probs(self, texts, images):
        probs = np.zeros((len(images), len(texts)))

        # Iterate through images in batches
        for i in range(0, len(images), self.config.batch_size):
            image_batch = images[i : i+self.config.batch_size]
            inputs = self.processor(text=texts, images=image_batch, return_tensors="pt", padding=True)

            # Move inputs to GPU
            for key, val in inputs.items():
                inputs[key] = val.to(self.config.device)

            outputs = self.model(**inputs)

            logits_per_image = outputs.logits_per_image
            batch_probs = logits_per_image.softmax(dim=1)
            batch_probs = batch_probs.cpu().numpy()

            probs[i : i+self.config.batch_size, :] = batch_probs

        return probs

    def init_detection(self, image, cat_names):
        w, h = image.size

        image_query = [image]
        if self.config.init_strat == 'segments':
            for curr_segments in range(2, self.config.num_segments):
                w_seg = int (w / curr_segments)
                h_seg = int (h / curr_segments)

                for i in range(curr_segments):
                    for j in range(curr_segments):
                        left, right = w_seg * i, w_seg * (i + 1)
                        top, bottom = h_seg * j, h_seg * (j + 1)

                        cropped_image = image.crop((left, top, right, bottom))
                        image_query.append(cropped_image)

        leftover_cats = set(cat_names)
        promising_cats = set()
        while True:
            text_query = [self.config.text_query_prepend + cat for cat in leftover_cats]

            init_probs = self._get_probs(text_query, image_query)
            init_probs = np.amax(init_probs, axis=0)

            for i, cat_name in enumerate(leftover_cats):
                if init_probs[i] > self.config.init_threshold:
                    promising_cats.add(cat_name)

            prev_leftover = leftover_cats.copy()
            leftover_cats = leftover_cats.difference(promising_cats)

            if not self.config.repeat_wo_best or prev_leftover==leftover_cats:
                break

        # Fill the negative categories list with all the leftover categories
        negative_cats = leftover_cats

        return list(promising_cats), list(negative_cats)

    def _generate_random_bboxes(self, img_width, img_height, num):
        lefts = np.random.uniform(0, img_width - MIN_BBOX_DIM, num).astype(int)
        tops = np.random.uniform(0, img_height - MIN_BBOX_DIM, num).astype(int)
        rights = np.random.uniform(lefts + MIN_BBOX_DIM, img_width, num).astype(int)
        downs = np.random.uniform(tops + MIN_BBOX_DIM, img_height, num).astype(int)

        return np.stack([lefts, tops, rights, downs], axis=-1).astype(int)

    def _crop_img(self, image, bbox):
        left, top, right, down = bbox[0], bbox[1], bbox[2], bbox[3]

        if self.crop:
            proc_image = image.crop((left, top, right, down))
        else:
            pixels = image.load()

            pixels[:top, :] = (0, 0, 0)
            pixels[top:down, :left] = (0, 0, 0)
            pixels[top:down, right:] = (0, 0, 0)
            pixels[down:, :] = (0, 0, 0)

        return proc_image

    def _random_detect(self, image, text_query, num_tries):
        bboxes = self._generate_random_bboxes(image.width, image.height, num_tries)

        confidence_scores = np.zeros(num_tries)

        #TODO remove this
        # Iterate through bboxes in batches
        for i in range(0, num_tries, self.batch_size):
            bbox_batch = bboxes[i:i+self.batch_size]
            image_batch = []
            for bbox in bbox_batch:
                proc_image = self._crop_img(image, bbox)
                image_batch.append(proc_image)

            # Run images through the model
            probs = self._get_probs(text_query, image_batch)
            #probs = np.zeros((len(bbox_batch), 2))
            confidence_scores[i:i+self.batch_size] = probs[:, 0]

        detections = []
        for i, score in enumerate(confidence_scores):
            if score > 0.5:
                detections.append({'bbox': bboxes[i].tolist(), 'score':score})
        
        return detections

    def detect(self, image, cat_names):

        self.crop = crop

        # Initial class filtering
        pass

        all_detections = []
        # Loop through all categories that may be present on the image
        for target_cat in tqdm(promising_cats):

            # Generate text query
            if text_strat == "one-negative":
                text_query = [target_cat, negative_text]
            elif text_strat == "negative-cats":
                text_query = [target_cat]
                text_query.extend(random.sample(negative_cats, min(num_other_cats, len(negative_cats))))
            elif text_strat == "promising-cats":
                text_query = [target_cat]
                text_query.extend(random.sample(promising_cats, min(num_other_cats, len(promising_cats))))
            else:
                raise ValueError(
                    f"Invalid text querying strategy: {text_strat}. "
                    "Must be one of [one-negative, negative-cats, promising-cats]")
            
            # Find bboxes
            if bbox_strat == "random":
                detections = self._random_detect(image, text_query, num_tries)
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
