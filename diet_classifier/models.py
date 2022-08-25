import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .config import DIETClassifierConfig

class IntentClassifier(nn.Module):
    def __init__(self, config: DIETClassifierConfig):
        super().__init__()
        # Rasa's embedding layer is actually a "dense embedding layer" which is just a Keras dense layer
        # equivalent to a PyTorch Linear layer.
        self.sentence_embed = nn.Linear(config.sentence_feature_dimension, config.embedding_dimension)
        self.label_embed = nn.Linear(config.num_intents, config.embedding_dimension)

    def forward(self, sentence_features: Tensor, label_features: Tensor):
        sentence_embedding = self.sentence_embed(sentence_features)
        label_embedding = self.label_embed(label_features)

        # dot product similarities
        similarities = torch.mm(sentence_embedding, label_embedding.t())

        return F.softmax(similarities, dim=1)
