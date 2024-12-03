from io import BytesIO
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
import torch
from PIL import Image
import logging
import os
from typing import List, cast
from fastapi import FastAPI
from pydantic import BaseModel
import base64
from colpali_engine.models import ColQwen2, ColQwen2Processor, ColPali, ColPaliProcessor
from transformers import BatchEncoding, BatchFeature, PreTrainedModel

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
logger.info(f"Detected device: {device}")


model_name = os.environ.get("MODEL_NAME", None)
if not model_name:
    raise ValueError("Environment variable MODEL_NAME not set")

model_class: type[PreTrainedModel]
processor_class: type[BaseVisualRetrieverProcessor]
if "colqwen2" in model_name:
    model_class = ColQwen2
    processor_class = ColQwen2Processor
elif "colpali" in model_name:
    model_class = ColPali
    processor_class = ColPaliProcessor
else:
    raise ValueError(f"Can not detect model family {model_name=}")


local_files_only = True


model = model_class.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    local_files_only=local_files_only,
).eval()

processor = cast(
    BaseVisualRetrieverProcessor,
    processor_class.from_pretrained(model_name, local_files_only=local_files_only),
)


app = FastAPI()

BatchEmbeddings = List[List[List[float]]]


class ProcessQueryResponseDto(BaseModel):
    embedding_batches: BatchEmbeddings


class ProcessQueryRequestDto(BaseModel):
    queries: List[str]


class ProcessImagesRequestDto(BaseModel):
    images: List[str]


class ScoreRequestDto(BaseModel):
    heystack_batch: BatchEmbeddings
    needle_batch: BatchEmbeddings


class ScoreResponseDto(BaseModel):
    scores: List[float]


def create_embeddings(batch: BatchFeature | BatchEncoding):
    # Forward pass
    with torch.no_grad():
        embeddings = model(**batch)
    return embeddings.cpu().float().numpy().tolist()


@app.post("/process-queries")
def process_queries(body: ProcessQueryRequestDto) -> ProcessQueryResponseDto:
    batch_queries = processor.process_queries(body.queries).to(model.device)
    return ProcessQueryResponseDto(embedding_batches=create_embeddings(batch_queries))


@app.post("/process-images")
def process_images(body: ProcessImagesRequestDto) -> ProcessQueryResponseDto:
    pil_images = [
        cast(Image.Image, Image.open(BytesIO(base64.b64decode(img_b64))))
        for img_b64 in body.images
    ]

    batch_queries = processor.process_images(pil_images).to(model.device)
    return ProcessQueryResponseDto(embedding_batches=create_embeddings(batch_queries))


@app.post("/score")
def calc_score(body: ScoreRequestDto) -> ScoreResponseDto:
    return ScoreResponseDto(
        scores=processor.score_multi_vector(
            torch.FloatTensor(body.heystack_batch), torch.FloatTensor(body.needle_batch)
        ).cpu().float().numpy().tolist()
    )


@app.get("/")
def read_root():
    return {"version": "1.0.0"}
