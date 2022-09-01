from typing import NamedTuple


class DIETClassifierConfig(NamedTuple):
    """
    Configuration for the DIET classifier.
    """
    # Model parameters
    num_intents: int
    sentence_feature_dimension: int = 1024
    embedding_dimension: int = 40
