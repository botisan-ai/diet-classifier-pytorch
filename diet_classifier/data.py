from typing import Dict, Any, List
from pathlib import Path
import yaml
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from jina import Flow
from docarray import DocumentArray, Document


class DIETClassifierDataModule(pl.LightningDataModule):
    def __init__(
        self,
        featurizer_host: str ='featurizer',
        featurizer_port: int = 8888,
        filename: str = 'nlu.yml',
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self.flow = Flow().add(host=featurizer_host, port=featurizer_port, external=True)
        self.filename = filename
        self.batch_size = batch_size
        self.read_nlu_file()

    def read_nlu_file(self):
        nlu_file = open(Path(self.filename).resolve(), 'r')
        nlu: Dict[str, Any] = yaml.load(nlu_file, Loader=yaml.Loader)
        self.nlu_intents: List[Dict[str, Any]] = nlu.get('nlu', [])
        self.num_intents = len(self.nlu_intents)
        self.label_data = torch.eye(len(self.nlu_intents))

    def setup(self, stage: str):
        self.flow.start()
        self.read_nlu_file()
        # one hot encoding for labels
        intent_data_raw = []
        for i, intent in enumerate(self.nlu_intents):
            examples: List[str] = intent.get('examples', [])
            example_docs = [Document(text=sentence) for sentence in examples]
            for doc in example_docs:
                intent_data_raw.append((doc, i))

        self.intent_dataset = []
        examples_da = DocumentArray([d[0] for d in intent_data_raw])
        features_da: DocumentArray = self.flow.post('/', inputs=examples_da, show_progress=True, request_size=10)

        for i, feature in enumerate(features_da):
            src = torch.from_numpy(np.array(feature.chunks[2].embedding))
            tgt = torch.tensor(intent_data_raw[i][1])
            self.intent_dataset.append((src, tgt))

    def train_dataloader(self):
        return DataLoader(self.intent_dataset, batch_size=self.batch_size, shuffle=True)

    def teardown(self, stage: str):
        self.flow.close()

if __name__ == '__main__':
    data_module = DIETClassifierDataModule()
    data_module.prepare_data()
