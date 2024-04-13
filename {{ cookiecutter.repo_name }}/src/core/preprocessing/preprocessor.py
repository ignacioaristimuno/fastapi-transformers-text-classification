from src.models.payload import TextsPayload
from src.settings import custom_logger


class Preprocessor:
    """Class for handling the preprocessing operations of the text data"""

    def __init__(self) -> None:
        self.logger = custom_logger(self.__class__.__name__)

    def preprocess_texts(self, payload: TextsPayload) -> TextsPayload:
        """Method for preprocessing a batch of text data"""

        return TextsPayload(
            texts=[self.preprocess_text(text) for text in payload.texts]
        )

    def preprocess_text(self, text):
        """Method for preprocessing a single text"""

        new_text = []
        for t in text.split(" "):
            t = "@user" if t.startswith("@") and len(t) > 1 else t
            t = "http" if t.startswith("http") else t
            new_text.append(t)
        return " ".join(new_text)
