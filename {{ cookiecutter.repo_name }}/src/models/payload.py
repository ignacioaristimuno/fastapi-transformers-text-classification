from pydantic import BaseModel
from typing import List

from src.models.texts import ClassifiedText


class TextsPayload(BaseModel):
    """Input payload for the classification endpoint"""

    texts: List[str]


class ResponseTextsPayload(BaseModel):
    """Response payload for the classification endpoint"""

    texts: List[ClassifiedText]
    model_id: str
