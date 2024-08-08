import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.multimodal import clip_score
from functools import partial

from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)


class DirectionalSimilarity(nn.Module):
    def __init__(self, tokenizer, text_encoder, image_processor, image_encoder, device):
        super().__init__()
        # self.tokenizer = tokenizer
        # self.text_encoder = text_encoder
        # self.image_processor = image_processor
        # self.image_encoder = image_encoder

        clip_id = "openai/clip-vit-large-patch14"
        tokenizer = CLIPTokenizer.from_pretrained(clip_id)
        text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to(device)
        image_processor = CLIPImageProcessor.from_pretrained(clip_id)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(device)

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.image_processor = image_processor
        self.image_encoder = image_encoder
        self.device = device

    def preprocess_image(self, image):
        image = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return {"pixel_values": image.to(self.device)}

    def tokenize_text(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"input_ids": inputs.input_ids.to(self.device)}

    def encode_image(self, image):
        preprocessed_image = self.preprocess_image(image)
        image_features = self.image_encoder(**preprocessed_image).image_embeds
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def encode_text(self, text):
        tokenized_text = self.tokenize_text(text)
        text_features = self.text_encoder(**tokenized_text).text_embeds
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def compute_directional_similarity(self, img_feat_one, img_feat_two, text_feat_one, text_feat_two):
        sim_direction = F.cosine_similarity(img_feat_two - img_feat_one, text_feat_two - text_feat_one)
        return sim_direction

    def forward(self, image_one, image_two, caption_one, caption_two):
        img_feat_one = self.encode_image(image_one)
        img_feat_two = self.encode_image(image_two)
        text_feat_one = self.encode_text(caption_one)
        text_feat_two = self.encode_text(caption_two)
        directional_similarity = self.compute_directional_similarity(
            img_feat_one, img_feat_two, text_feat_one, text_feat_two
        )
        return directional_similarity