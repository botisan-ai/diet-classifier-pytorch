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

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        sentence_features, target = batch
        similarities = self.forward(sentence_features)
        loss = F.cross_entropy(similarities, target)
        self.log('train_loss', loss)
        return loss
