from typing import Tuple
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
        self.loss = nn.CrossEntropyLoss()

    def forward(self, sentence_features: Tensor) -> Tuple[Tensor, Tensor]:
        label_features = self._get_label_features()
        return self.intent_classifier(
            sentence_features,
            label_features,
        )

    def predict(self, sentence_features: Tensor):
        sentence_embeddings, label_embeddings = self.forward(sentence_features)
        similarities = torch.mm(sentence_embeddings, label_embeddings.t())
        return F.softmax(similarities, dim=-1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        input_features, target_labels = batch
        input_embeddings, label_embeddings = self.forward(input_features)
        target_embeddings = label_embeddings[target_labels]

        loss, accuracy = self._calculate_loss_and_accuracy(
            input_embeddings,
            label_embeddings,
            target_embeddings,
            target_labels,
        )

        self.log('train_loss', loss)
        self.log('train_acc', accuracy)

        return loss

    def _calculate_loss_and_accuracy(
        self,
        input_embeddings: Tensor,
        label_embeddings: Tensor,
        target_embeddings: Tensor,
        target_labels: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        (
            pos_input_embeddings,
            pos_label_embeddings,
            neg_input_embeddings,
            neg_label_embeddings,
            input_bad_negatives,
            label_bad_negatives,
        ) = self._sample_negatives(
            input_embeddings,
            label_embeddings,
            target_embeddings,
            target_labels,
        )

        # calculate similarities
        (
            sim_pos,
            sim_neg_input_label,
            sim_neg_label_label,
            sim_neg_input_input,
            sim_neg_label_input,
        ) = self._calculate_similarity(
            pos_input_embeddings,
            pos_label_embeddings,
            neg_input_embeddings,
            neg_label_embeddings,
            input_bad_negatives,
            label_bad_negatives,
        )

        accuracy = self._calculate_accuracy(sim_pos, sim_neg_input_label)

        loss = self._calculate_loss(
            sim_pos,
            sim_neg_input_label,
            sim_neg_label_label,
            sim_neg_input_input,
            sim_neg_label_input,
        )

        return loss, accuracy

    def _sample_negatives(
        self,
        input_embeddings: Tensor,
        label_embeddings: Tensor,
        target_embeddings: Tensor,
        target_labels: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        pos_input_embeddings = input_embeddings.unsqueeze(-2)
        pos_label_embeddings = target_embeddings.unsqueeze(-2)

        # sample negative inputs
        neg_input_embeddings, input_bad_negatives = self._get_negatives(
            input_embeddings,
            target_labels,
            target_labels,
        )
        # sample negative labels
        neg_label_embeddings, label_bad_negatives = self._get_negatives(
            label_embeddings,
            self._get_all_labels(),
            target_labels,
        )

        return (
            pos_input_embeddings,
            pos_label_embeddings,
            neg_input_embeddings,
            neg_label_embeddings,
            input_bad_negatives,
            label_bad_negatives,
        )

    def _get_negatives(
        self,
        embeddings: Tensor,
        labels: Tensor,
        target_labels: Tensor,
    ):
        # batch_flatten
        embeddings_flattened = embeddings.view(-1, embeddings.shape[-1])
        labels_flattened = labels.view(-1, labels.shape[-1])
        target_labels_flattened = target_labels.view(-1, target_labels.shape[-1])

        total_candidates = embeddings_flattened.shape[0]
        target_size = target_labels_flattened.shape[1]

        # assign random indices as negative labels
        negative_indices = torch.randint(
            total_candidates,
            (target_size, self.config.num_negative_samples),
        )

        negative_embeddings = embeddings_flattened[negative_indices]
        negative_labels = labels_flattened.t()[negative_indices]

        bad_negatives = torch.eq(negative_labels, target_labels_flattened.t().unsqueeze(-1)).all(dim=-1)

        return negative_embeddings, bad_negatives

    def _calculate_similarity(
        self,
        pos_input_embeddings: Tensor,
        pos_label_embeddings: Tensor,
        neg_input_embeddings: Tensor,
        neg_label_embeddings: Tensor,
        input_bad_negatives: Tensor,
        label_bad_negatives: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        neg_infinity = torch.tensor(-1e9)

        # inner dot product
        sim_pos = torch.sum(pos_input_embeddings * pos_label_embeddings, dim=-1)
        sim_neg_input_label = torch.sum(pos_input_embeddings * neg_label_embeddings, dim=-1) + neg_infinity * label_bad_negatives
        sim_neg_label_label = torch.sum(pos_label_embeddings * neg_label_embeddings, dim=-1) + neg_infinity * label_bad_negatives
        sim_neg_input_input = torch.sum(pos_input_embeddings * neg_input_embeddings, dim=-1) + neg_infinity * input_bad_negatives
        sim_neg_label_input = torch.sum(pos_label_embeddings * neg_input_embeddings, dim=-1) + neg_infinity * input_bad_negatives

        return sim_pos, sim_neg_input_label, sim_neg_label_label, sim_neg_input_input, sim_neg_label_input

    def _calculate_accuracy(
        self,
        sim_pos: Tensor,
        sim_neg: Tensor,
    ):
        max_all_sim = torch.cat([sim_pos, sim_neg], dim=-1).max(dim=-1).values
        return torch.eq(max_all_sim, sim_pos.squeeze(-1)).float().mean()

    def _calculate_loss(
        self,
        sim_pos: Tensor,
        sim_neg_input_label: Tensor,
        sim_neg_label_label: Tensor,
        sim_neg_input_input: Tensor,
        sim_neg_label_input: Tensor,
    ) -> Tensor:
        # concatenate all similarities
        logits = torch.cat([
            sim_pos,
            sim_neg_input_label,
            sim_neg_label_label,
            sim_neg_input_input,
            sim_neg_label_input,
        ], dim=-1)

        # create target labels, since the positive samples are always the first element
        # create all zeros
        target_labels = torch.zeros(logits.shape[0], dtype=torch.long)

        return self.loss(logits, target_labels)

    def _get_all_labels(self):
        return torch.arange(0, self.config.num_intents)

    def _get_label_features(self):
        return torch.eye(self.config.num_intents)
