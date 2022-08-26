from jina import Flow
from docarray import DocumentArray, Document

from executor import DIETClassifierExecutor

f = Flow().add(
    uses='jinahub+docker://ConveRTFeaturizer/latest'
).add(
    uses=DIETClassifierExecutor, uses_with={ 'model_path': './lightning_logs/version_4/checkpoints/epoch=999-step=1000.ckpt' }
)

with f:
    inputs = DocumentArray([Document(text='Naw man')])
    outputs: DocumentArray = f.post('/', inputs)
    for doc in outputs:
        doc.summary()
