from jina import Flow
from docarray import DocumentArray, Document
from pathlib import Path
import yaml
from typing import Dict, Any, List

from executor import DIETClassifierExecutor

f = Flow(
    prefetch=1,
).add(
    # host='paddlenlp.default.api.senses.chat',
    # port=80,
    host='featurizer',
    port=8888,
    external=True,
).add(
    uses=DIETClassifierExecutor,
    uses_with={
        'nlu_filename': 'intents202208.yml',
        'model_path': './lightning_logs/version_0/checkpoints/epoch=999-step=10000.ckpt',
        # 'sentence_feature_dimension': 384,
        'sentence_feature_dimension': 768,
    },
)

nlu_file = open(Path('intents202208.yml').resolve(), 'r')
nlu: Dict[str, Any] = yaml.load(nlu_file, Loader=yaml.Loader)
nlu_intents: List[Dict[str, Any]] = nlu.get('nlu', [])

with f:
    docs = []
    for i, intent in enumerate(nlu_intents):
        examples: List[str] = intent.get('examples', [])
        example_docs = [Document(text=sentence) for sentence in examples]
        docs.extend(example_docs)
    inputs = DocumentArray(docs)
    outputs: DocumentArray = f.post('/', inputs, show_progress=True, request_size=10)
    outputs.save_binary('part-2-embeddings.bin', protocol='pickle-array', compress='gzip')
    # for doc in outputs:
    #     doc.summary()
