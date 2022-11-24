from typing import Any, Dict, List
import yaml
from pathlib import Path
from jina import Executor, requests
from docarray import DocumentArray, Document
from docarray.score import NamedScore
import torch

from diet_classifier.config import DIETClassifierConfig
from diet_classifier.classifier import DIETClassifier

class DIETClassifierExecutor(Executor):
    def __init__(self, nlu_filename='nlu.yml', model_path='./lightning_logs', sentence_feature_dimension: int = 1024, **kwargs):
        super().__init__(**kwargs)
        self.nlu_filename = nlu_filename
        self.read_nlu_file()
        config = DIETClassifierConfig(num_intents=self.num_intents, sentence_feature_dimension=sentence_feature_dimension)
        self.model = DIETClassifier.load_from_checkpoint(Path(model_path).resolve(), config=config)

    def read_nlu_file(self):
        nlu_file = open(Path(self.nlu_filename).resolve(), 'r')
        nlu: Dict[str, Any] = yaml.load(nlu_file, Loader=yaml.Loader)
        self.nlu_intents: List[Dict[str, Any]] = nlu.get('nlu', [])
        self.num_intents = len(self.nlu_intents)

    @requests
    def request(self, docs: DocumentArray, **kwargs) -> DocumentArray:
        embeddings = docs['@c[2]'].embeddings
        similarities = self.model.predict(torch.tensor(embeddings))
        for i, doc in enumerate(docs):
            doc.embedding = similarities[i].detach().numpy()
            for j in range(self.num_intents):
                score = similarities[i].detach().numpy()[j]
                intent = Document(text=self.nlu_intents[j]['intent'], modality='intent')
                intent.scores['confidence'] = NamedScore(value=score, description='confidence')
                doc.matches.append(intent)
            doc.matches = sorted(doc.matches, key=lambda d: d.scores['confidence'].value, reverse=True)
        return docs
