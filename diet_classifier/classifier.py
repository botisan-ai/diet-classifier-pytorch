import torch
from torch import optim, nn, Tensor
import torch.nn.functional as F
import pytorch_lightning as pl

from .config import DIETClassifierConfig
from .models import IntentClassifier

class DIETClassifier(pl.LightningModule):
    def __init__(self, config: DIETClassifierConfig):
        super().__init__()
        self.config = config
        self.intent_classifier = IntentClassifier(config)

    def forward(self, sentence_features: Tensor):
        label_features = torch.eye(self.config.num_intents)
        return self.intent_classifier(sentence_features, label_features)

    def predict(self, sentence_features: Tensor):
        sentence_embeddings, label_embeddings = self.forward(sentence_features)
        similarities = torch.mm(sentence_embeddings, label_embeddings.t())
        return {
            'similarities': F.softmax(similarities, dim=-1),
            'sentence_embeddings': sentence_embeddings,
        }

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        sentence_features, target = batch
        similarities = self.forward(sentence_features)
        loss = F.cross_entropy(similarities, target)
        self.log('train_loss', loss)
        return loss
