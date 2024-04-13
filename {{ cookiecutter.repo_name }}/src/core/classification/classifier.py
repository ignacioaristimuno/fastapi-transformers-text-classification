import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from typing import List

from src.models.labels import ClassificationLabel
from src.models.texts import ClassifiedText, ScoresMetadata
from src.models.payload import TextsPayload, ResponseTextsPayload
from src.settings import custom_logger


class Classifier:
    """Class for handling the text classification operations."""

    def __init__(self, model_id: str, batch_size: int):
        self.logger = custom_logger(self.__class__.__name__)
        self.model_id = model_id
        self.batch_size = batch_size

        # Model loading
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        self.load_model()

    def load_model(self):
        """Method for loading the model and tokenizer from the Hugging Face model hub."""

        self.logger.info(f"Loading model {self.model_id}")
        self.model_config = AutoConfig.from_pretrained(self.model_id)
        self.num_labels = len(self.model_config.id2label)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, num_labels=self.num_labels
        ).to(self.device)
        self.logger.info(f"Model {self.model_id} loaded successfully!")

    def _create_batches(self, texts: List[str]):
        """Yield successive n-sized chunks from a list of texts."""

        for i in range(0, len(texts), self.batch_size):
            yield texts[i : i + self.batch_size]

    def predict(self, payload: TextsPayload) -> ResponseTextsPayload:
        """Method for predicting the class of a batch of texts."""

        results = []
        for batch_texts in self._create_batches(payload.texts):
            # Model inference
            encoded_input = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            outputs = self.model(**encoded_input)

            # Process the outputs for the entire batch at once
            scores = outputs.logits.detach().numpy()
            scores = torch.softmax(torch.tensor(scores), dim=-1).numpy()
            ranking = np.argsort(scores, axis=-1)[:, ::-1]

            # Format output
            for i, text in enumerate(batch_texts):
                label_scores = {
                    ClassificationLabel(
                        self.model_config.id2label[ranking[i][j]]
                    ): scores[i][ranking[i][j]]
                    for j in range(scores.shape[1])
                }

                result = ClassifiedText(
                    text=text,
                    label=ClassificationLabel(
                        self.model_config.id2label[ranking[i][0]]
                    ),
                    score=scores[i][ranking[i][0]],
                    metadata=ScoresMetadata(scores=label_scores),
                )
                results.append(result)

        return ResponseTextsPayload(
            texts=results,
            model_id=self.model_id,
        )
