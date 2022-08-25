from ast import List
from typing import Any, Dict
import yaml
from pathlib import Path
from jina import Executor, requests
from docarray import DocumentArray, Document
import torch

from diet_classifier.config import DIETClassifierConfig
from diet_classifier.classifier import DIETClassifier

class DIETClassifierExecutor(Executor):
    def __init__(self, nlu_filename='nlu.yml', model_path='./lightning_logs', **kwargs):
        super().__init__(**kwargs)
        self.nlu_filename = nlu_filename
        self.read_nlu_file()
        config = DIETClassifierConfig(num_intents=self.num_intents)
        self.model = DIETClassifier.load_from_checkpoint(Path(model_path).resolve(), config=config)

    def read_nlu_file(self):
        nlu_file = open(Path(self.nlu_filename).resolve(), 'r')
        nlu: Dict[str, Any] = yaml.load(nlu_file, Loader=yaml.Loader)
        self.nlu_intents: List[Dict[str, Any]] = nlu.get('nlu', [])
        self.num_intents = len(self.nlu_intents)

    @requests
    def request(self, docs: DocumentArray, **kwargs) -> DocumentArray:
        similarities = self.model(torch.tensor(docs.embeddings))
        for i, doc in enumerate(docs):
            doc.embedding = similarities[i].detach().numpy()
            for j in range(self.num_intents):
                score = similarities[i].detach().numpy()[j]
                intent = Document(text=self.nlu_intents[j]['intent'], confidence=float(score))
                intent.summary()
                doc.chunks.append(intent)
        return docs
