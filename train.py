
import pytorch_lightning as pl

from diet_classifier.config import DIETClassifierConfig
from diet_classifier.classifier import DIETClassifier
from diet_classifier.data import DIETClassifierDataModule


datamodule = DIETClassifierDataModule(filename='nlu_zh.yml', batch_size=32)
config = DIETClassifierConfig(
    num_intents=datamodule.num_intents,
    # sentence_feature_dimension=384,
    sentence_feature_dimension=768,
)
model = DIETClassifier(config)

trainer = pl.Trainer(max_epochs=1000, log_every_n_steps=50)
trainer.fit(model=model, datamodule=datamodule)
