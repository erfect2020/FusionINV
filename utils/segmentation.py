from typing import Tuple, List

import nltk
import numpy as np
import torch
from sklearn.cluster import KMeans

from constants import VIS_INDEX, IR_INDEX

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

"""
Self-segmentation technique taken from Prompt Mixing: https://github.com/orpatashnik/local-prompt-mixing
"""

class Segmentor:

    def __init__(self, prompt: str, object_nouns: List[str], num_segments: int = 5, res: int = 32):
        self.prompt = prompt
        self.num_segments = num_segments
        self.resolution = res
        self.object_nouns = object_nouns
        tokenized_prompt = nltk.word_tokenize(prompt)
        forbidden_words = [word.upper() for word in ["photo", "image", "picture"]]
        self.nouns = [(i, word) for (i, (word, pos)) in enumerate(nltk.pos_tag(tokenized_prompt))
                      if pos[:2] == 'NN' and word.upper() not in forbidden_words]

    def update_attention(self, attn, is_cross):
        res = int(attn.shape[2] ** 0.5)
        if is_cross:
            if res == 16:
                self.cross_attention_32 = attn
            elif res == 32:
                self.cross_attention_64 = attn
            elif res == 64:
                self.cross_attention_128 = attn
        else:
            if res == 32:
                self.self_attention_32 = attn
            elif res == 64:
                self.self_attention_64 = attn
            elif res == 128:
                self.self_attention_128 = attn

    def __call__(self, *args, **kwargs): #meanlessness
        clusters = self.cluster()
        cluster2noun = self.cluster2noun(clusters)
        return cluster2noun

    def cluster(self, res: int = 32):
        np.random.seed(1)
        self_attn = self.self_attention_32 if res == 32 else self.self_attention_64

        vis_attn = self_attn[VIS_INDEX].mean(dim=0).cpu().numpy()
        vis_kmeans = KMeans(n_clusters=self.num_segments, n_init=10).fit(vis_attn)
        vis_clusters = vis_kmeans.labels_.reshape(res, res)

        ir_attn = self_attn[IR_INDEX].mean(dim=0).cpu().numpy()
        ir_kmeans = KMeans(n_clusters=self.num_segments, n_init=10).fit(ir_attn)
        ir_clusters = ir_kmeans.labels_.reshape(res, res)

        return vis_clusters, ir_clusters

    def cluster2noun(self, clusters, cross_attn, attn_index):
        result = {}
        res = int(cross_attn.shape[2] ** 0.5)
        nouns_indices = [index for (index, word) in self.nouns]
        cross_attn = cross_attn[attn_index].mean(dim=0).reshape(res, res, -1)
        nouns_maps = cross_attn.cpu().numpy()[:, :, [i + 1 for i in nouns_indices]]
        normalized_nouns_maps = np.zeros_like(nouns_maps).repeat(2, axis=0).repeat(2, axis=1)
        for i in range(nouns_maps.shape[-1]):
            curr_noun_map = nouns_maps[:, :, i].repeat(2, axis=0).repeat(2, axis=1)
            normalized_nouns_maps[:, :, i] = (curr_noun_map - np.abs(curr_noun_map.min())) / curr_noun_map.max()
        
        max_score = 0
        all_scores = []
        for c in range(self.num_segments):
            cluster_mask = np.zeros_like(clusters)
            cluster_mask[clusters == c] = 1
            score_maps = [cluster_mask * normalized_nouns_maps[:, :, i] for i in range(len(nouns_indices))]
            scores = [score_map.sum() / cluster_mask.sum() for score_map in score_maps]
            all_scores.append(max(scores))
            max_score = max(max(scores), max_score)

        all_scores.remove(max_score)
        mean_score = sum(all_scores) / len(all_scores)

        for c in range(self.num_segments):
            cluster_mask = np.zeros_like(clusters)
            cluster_mask[clusters == c] = 1
            score_maps = [cluster_mask * normalized_nouns_maps[:, :, i] for i in range(len(nouns_indices))]
            scores = [score_map.sum() / cluster_mask.sum() for score_map in score_maps]
            result[c] = self.nouns[np.argmax(np.array(scores))] if max(scores) > 1.4 * mean_score else "BG"

        return result

    def create_mask(self, clusters, cross_attention, attn_index):
        cluster2noun = self.cluster2noun(clusters, cross_attention, attn_index)
        mask = clusters.copy()
        obj_segments = [c for c in cluster2noun if cluster2noun[c][1] in self.object_nouns]
        for c in range(self.num_segments):
            mask[clusters == c] = 1 if c in obj_segments else 0
        return torch.from_numpy(mask).to("cuda")

    def get_object_masks(self) -> Tuple[torch.Tensor]:
        clusters_vis_32, clusters_ir_32 = self.cluster(res=32)
        clusters_vis_64, clusters_ir_64 = self.cluster(res=64)
        # clusters_vis_128, clusters_ir_128 = self.cluster(res=128)

        mask_ir_32 = self.create_mask(clusters_ir_32, self.cross_attention_32, IR_INDEX)
        mask_vis_32 = self.create_mask(clusters_vis_32, self.cross_attention_32, VIS_INDEX)
        mask_ir_64 = self.create_mask(clusters_ir_64, self.cross_attention_64, IR_INDEX)
        mask_vis_64 = self.create_mask(clusters_vis_64, self.cross_attention_64, VIS_INDEX)
        # mask_ir_128 = self.create_mask(clusters_ir_128, self.cross_attention_128, IR_INDEX)
        # mask_vis_128 = self.create_mask(clusters_vis_128, self.cross_attention_128, VIS_INDEX)

        return mask_ir_32, mask_vis_32, mask_ir_64, mask_vis_64 #, mask_ir_128, mask_vis_128
