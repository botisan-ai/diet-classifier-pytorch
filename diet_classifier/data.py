from typing import Dict, Any, List
from pathlib import Path
import yaml
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from jina import Flow
from docarray import DocumentArray, Document


class DIETClassifierDataModule(pl.LightningDataModule):
    def __init__(self, convert_featurizer_host: str = 'jinahub+docker://ConveRTFeaturizer/latest', filename: str = 'nlu.yml', batch_size: int = 32) -> None:
        super().__init__()
        self.flow = Flow().add(uses=convert_featurizer_host)
        self.filename = filename
        self.batch_size = batch_size
        self.read_nlu_file()

    def read_nlu_file(self):
        nlu_file = open(Path(self.filename).resolve(), 'r')
        nlu: Dict[str, Any] = yaml.load(nlu_file, Loader=yaml.Loader)
        self.nlu_intents: List[Dict[str, Any]] = nlu.get('nlu', [])
        self.num_intents = len(self.nlu_intents)
        self.label_data = torch.eye(len(self.nlu_intents))

    def prepare_data(self):
        self.read_nlu_file()
        # one hot encoding for labels
        self.intent_dataset = []
        with self.flow:
            for i, intent in enumerate(self.nlu_intents):
                examples: List[str] = intent.get('examples', [])
                example_da = DocumentArray([Document(text=sentence) for sentence in examples])
                features_da: DocumentArray = self.flow.post('/', inputs=example_da, show_progress=True)
                for feature in features_da:
                    self.intent_dataset.append((torch.from_numpy(feature.embedding), self.label_data[i]))

    def train_dataloader(self):
        return DataLoader(self.intent_dataset, batch_size=self.batch_size, shuffle=True)


if __name__ == '__main__':
    data_module = DIETClassifierDataModule()
    data_module.prepare_data()
