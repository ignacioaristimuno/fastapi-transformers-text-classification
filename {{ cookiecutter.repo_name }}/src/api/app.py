from contextlib import asynccontextmanager
from fastapi import FastAPI
import os
import sys
import uvicorn

sys.path.append(os.getcwd())

from src.api.routers import init_routers
from src.core.classification import Classifier
from src.core.preprocessing import Preprocessor
from src.settings import SettingsManager


# Context manager for loading classes on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Function for loading the classifier, preprocessor and settings on startup"""

    settings = SettingsManager()
    preprocessor = Preprocessor()
    classifier = Classifier(
        model_id=settings.MODEL_ID,
        batch_size=settings.BATCH_SIZE,
    )
    yield {"settings": settings, "preprocessor": preprocessor, "classifier": classifier}


# Create API
app = FastAPI(
    title="{{cookiecutter.project_name}}", debug=True, version="0.1", lifespan=lifespan
)

# Load routers
init_routers(app)


# Run API
if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8080, reload=True)
