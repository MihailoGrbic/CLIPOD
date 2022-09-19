import random

import matplotlib.pyplot as plt
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torch
from tqdm import tqdm

from object_detector_config import CLIPObjectDetectorConfig
from util import iou, fix_bbox

MIN_BBOX_DIM = 10


class CLIPObjectDetector:
    def __init__(self, model_name, device=None, config={}):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        if isinstance(config, str):
            self.config = CLIPObjectDetectorConfig.from_file(config)
        elif isinstance(config, dict):
            self.config = CLIPObjectDetectorConfig(config)
        elif isinstance(config, CLIPObjectDetectorConfig):
            self.config = config
        else:
            raise ValueError(f"Unknown object type {type(config)} for config")

        self.model.to(self.device).eval()

    def _get_probs(self, texts, images):
        probs = np.zeros((len(images), len(texts)))

        # Iterate through images in batches
        with torch.no_grad():
            for i in range(0, len(images), self.config.batch_size):
                image_batch = images[i: i + self.config.batch_size]
                inputs = self.processor(text=texts, images=image_batch, return_tensors='pt', padding=True)

                # Move inputs to GPU
                for key, val in inputs.items():
                    inputs[key] = val.to(self.device)

                outputs = self.model(**inputs)

                logits_per_image = outputs.logits_per_image
                batch_probs = logits_per_image.softmax(dim=1)
                batch_probs = batch_probs.cpu().numpy()

                probs[i: i + self.config.batch_size, :] = batch_probs

        return probs

    def init_detection(self, image, cat_names, return_probs=False):
        w, h = image.size

        image_prompt = [image]
        if self.config.init_strat == 'segments':
            for curr_segments in range(2, self.config.init_settings.num_segments + 1):
                w_seg = int(w / curr_segments)
                h_seg = int(h / curr_segments)

                for i in range(curr_segments):
                    for j in range(curr_segments):
                        left, right = w_seg * i, w_seg * (i + 1)
                        top, bottom = h_seg * j, h_seg * (j + 1)

                        cropped_image = self._crop_img(image, (left, top, right, bottom))
                        image_prompt.append(cropped_image)

        leftover_cats = cat_names.copy()
        promising_cats = []
        text_prompt = [self.config.text_prompt_prepend + cat for cat in cat_names]
        init_probs = self._get_probs(text_prompt, image_prompt)
        probs = init_probs.copy()

        for step in range(self.config.init_settings.max_repeat_steps):
            probs_max = np.amax(probs, axis=0)
            to_delete = []
            leftover_len = len(leftover_cats)
            for i in range(leftover_len - 1, -1, -1):
                if probs_max[i] > self.config.init_settings.threshold:
                    promising_cats.append(leftover_cats.pop(i))
                    to_delete.append(i)

            if not self.config.init_settings.repeat_wo_best or len(to_delete) == 0:
                break

            # Remove probs that correspond to detected classes
            probs = np.delete(probs, to_delete, axis=1)
            # Normalize the probs to sum up to 1
            probs = np.divide(probs.T, np.sum(probs, axis=1)).T

        negative_cats = leftover_cats

        if return_probs:
            return promising_cats, negative_cats, init_probs
        return promising_cats, negative_cats

    def _generate_random_bboxes(self, img_width, img_height, num):
        lefts = np.random.uniform(0, img_width - MIN_BBOX_DIM, num).astype(int)
        tops = np.random.uniform(0, img_height - MIN_BBOX_DIM, num).astype(int)
        rights = np.random.uniform(lefts + MIN_BBOX_DIM, img_width, num).astype(int)
        bottoms = np.random.uniform(tops + MIN_BBOX_DIM, img_height, num).astype(int)

        return np.stack([lefts, tops, rights, bottoms], axis=-1).astype(int)

    def _crop_img(self, image, bbox):
        left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]

        if self.config.crop:
            proc_image = image.crop((left, top, right, bottom))
            assert proc_image.width == right - left
            assert proc_image.height == bottom - top
        else:
            proc_image = image.copy()
            pixels = proc_image.load()
            for x in range(proc_image.width):
                for y in range(proc_image.height):
                    if x < left or x > right or y < top or y > bottom:
                        pixels[x, y] = (0, 0, 0)

        return proc_image

    def _random_detect(self, image, text_prompt):
        num_tries = self.config.detection_settings.num_tries
        bboxes = self._generate_random_bboxes(image.width, image.height, num_tries)

        image_prompt = []
        for bbox in bboxes:
            fix_bbox(bbox, image.width, image.height, MIN_BBOX_DIM)
            proc_image = self._crop_img(image, bbox)
            image_prompt.append(proc_image)

        confidence_scores = self._get_probs(text_prompt, image_prompt)[:, 0]

        detections = []
        for i, score in enumerate(confidence_scores):
            if score > 0.5:
                detections.append({'bbox': bboxes[i].tolist(), 'score': score})

        return detections

    def _beam_search(self, image, text_prompt):
        settings = self.config.detection_settings
        num_segments = settings.num_segments

        # Roughly locate objects before performing beam_search
        w_seg = int(image.width / num_segments)
        h_seg = int(image.height / num_segments)
        image_prompt = []
        for i in range(num_segments):
            for j in range(num_segments):
                left, right = w_seg * i, w_seg * (i + 1)
                top, bottom = h_seg * j, h_seg * (j + 1)

                cropped_image = self._crop_img(image, (left, top, right, bottom))
                image_prompt.append(cropped_image)

        init_confidence_scores = self._get_probs(text_prompt, image_prompt)[:, 0]

        detections = []
        for i in range(num_segments):
            for j in range(num_segments):
                if init_confidence_scores[i * num_segments + j] > settings.start_threshold:
                    # Start beam search
                    left, right = w_seg * i, w_seg * (i + 1)
                    top, bottom = h_seg * j, h_seg * (j + 1)
                    prev_score = init_confidence_scores[i * num_segments + j]
                    population = [[left, top, right, bottom]]
                    scores = [prev_score]

                    # Main beam-search loop
                    end_steps = 0
                    for step in range(settings.max_steps):
                        new_population = []
                        for x in population:
                            new_population.extend((x + np.random.randint(-settings.new_range,
                                                  settings.new_range, (settings.num_new, 4))).tolist())

                        image_prompt = []
                        for bbox in new_population:
                            fix_bbox(bbox, image.width, image.height, MIN_BBOX_DIM)
                            proc_image = self._crop_img(image, bbox)
                            image_prompt.append(proc_image)

                        new_scores = self._get_probs(text_prompt, image_prompt)[:, 0]

                        population.extend(new_population)
                        scores.extend(new_scores)

                        scores, population = zip(*sorted(zip(scores, population), reverse=True))

                        population = list(population)[:settings.num_saved]
                        scores = list(scores)[:settings.num_saved]

                        if scores[0] - prev_score < 0.001:
                            end_steps += 1
                        else:
                            end_steps = 0
                            prev_score = scores[0]

                        if end_steps >= settings.patience:

                            # img = image.crop(population[0])
                            # img.show()
                            # print(text_prompt[0])
                            break

                        #print(f"{scores[0]}  {end_steps}")
                    # print()
                    # print(f"{text_prompt[0]}  {step} {end_steps}")
                    if scores[0] > 0.5:
                        detections.append({'bbox': population[0], 'score': scores[0]})

        return detections

    def detect(self, image, cat_names):
        # Initial class filtering
        promising_cats, negative_cats = self.init_detection(image, cat_names)

        all_detections = []
        # Loop through all categories that may be present on the image
        for target_cat in tqdm(promising_cats):
            # Generate text prompt
            if self.config.detection_text_strat == 'negative-text':
                text_prompt = [self.config.text_prompt_prepend + target_cat]
                text_prompt.extend(self.config.detection_text_settings.negative_text)

            elif self.config.detection_text_strat == 'negative-cats':
                text_prompt = [target_cat]
                text_prompt.extend(
                    random.sample(
                        negative_cats, min(self.config.detection_text_settings.num_other_cats, len(negative_cats))))
                for i, cat_name in enumerate(text_prompt):
                    text_prompt[i] = self.config.text_prompt_prepend + cat_name

            elif self.config.detection_text_strat == 'promising-cats':
                text_prompt = [target_cat]
                text_prompt.extend(
                    random.sample(
                        promising_cats, min(self.config.detection_text_settings.num_other_cats, len(promising_cats))))

                # Remove copy of target_cat if present
                for i in range(1, len(text_prompt)):
                    if text_prompt[i] == target_cat:
                        del text_prompt[i]
                        break

                for i, cat_name in enumerate(text_prompt):
                    text_prompt[i] = self.config.text_prompt_prepend + cat_name

            elif self.config.detection_text_strat == 'all-cats':
                text_prompt = [target_cat]
                text_prompt.extend(cat_names)

                # Remove copy of target_cat if present
                for i in range(1, len(text_prompt)):
                    if text_prompt[i] == target_cat:
                        del text_prompt[i]
                        break

                for i, cat_name in enumerate(text_prompt):
                    text_prompt[i] = self.config.text_prompt_prepend + cat_name
            else:
                raise ValueError(
                    f"Unknown detection text prompting strategy: {self.config.detection_text_strat}. "
                    "Must be one of [negative-text, negative-cats, promising-cats, all-cats]")

            # Find bboxes
            if self.config.detection_strat == 'random':
                detections = self._random_detect(image, text_prompt)
            elif self.config.detection_strat == 'beam-search':
                detections = self._beam_search(image, text_prompt)
            else:
                raise ValueError(
                    f"Unknown detection strategy: {self.config.detection_strat}. "
                    "Must be one of [random, beam-search]")

            # Non-maximum suppression
            detections = sorted(detections, key=lambda x: x['score'], reverse=True)
            final_detections = []
            while len(detections) > 0:
                final_det = detections.pop(-1)
                final_detections.append(final_det)
                detections = [
                    x for x in detections 
                    if not iou(x['bbox'], final_det['bbox']) > self.config.detection_settings.nms_thresh]

            # Add field for category name
            for detection in final_detections:
                detection['cat_name'] = target_cat

            all_detections.extend(final_detections)

        return all_detections
