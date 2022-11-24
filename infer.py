from jina import Flow
from docarray import DocumentArray, Document

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

with f:
    inputs = DocumentArray([Document(text='大家好！')])
    outputs: DocumentArray = f.post('/', inputs)
    for doc in outputs:
        doc.summary()
