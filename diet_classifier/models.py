from typing import Tuple
from torch import nn, Tensor

from .config import DIETClassifierConfig

class IntentClassifier(nn.Module):
    def __init__(self, config: DIETClassifierConfig):
        super().__init__()
        # Rasa's embedding layer is actually a "dense embedding layer" which is just a Keras dense layer
        # equivalent to a PyTorch Linear layer.
        self.sentence_embed = nn.Linear(config.sentence_feature_dimension, config.embedding_dimension)
        self.label_embed = nn.Linear(config.num_intents, config.embedding_dimension)

    def forward(self, sentence_features: Tensor, label_features: Tensor) -> Tuple[Tensor, Tensor]:
        sentence_embeddings = self.sentence_embed(sentence_features)
        label_embeddings = self.label_embed(label_features)

        return sentence_embeddings, label_embeddings
