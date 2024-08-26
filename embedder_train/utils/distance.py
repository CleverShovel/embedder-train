import torch
from sentence_transformers.util import dot_score


def dot_distance(embeddings: torch.Tensor) -> torch.Tensor:
    return 1 - dot_score(embeddings, embeddings)