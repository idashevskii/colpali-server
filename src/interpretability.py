import io
from typing import cast
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
import matplotlib.pyplot as plt
import torch
from colpali_engine.interpretability import (
    get_similarity_maps_from_embeddings,
    plot_similarity_map,
)
from colpali_engine.models import ColQwen2Processor
from PIL import Image
from transformers import PreTrainedModel
from math import ceil


def create_interpret_image(
    image: Image.Image,
    query: str,
    model: PreTrainedModel,
    processor: BaseVisualRetrieverProcessor,
) -> Image.Image:
    processor = cast(ColQwen2Processor, processor)

    # Preprocess inputs
    batch_images = processor.process_images([image]).to(model.device)
    batch_queries = processor.process_queries([query]).to(model.device)

    # Forward passes
    with torch.no_grad():
        image_embeddings = model.forward(**batch_images)
        query_embeddings = model.forward(**batch_queries)

    # Get the number of image patches
    n_patches = processor.get_n_patches(
        image_size=image.size,
        patch_size=model.patch_size,
        spatial_merge_size=model.spatial_merge_size,
    )

    # print(f"Number of image patches: {n_patches}")

    # Get the tensor mask to filter out the embeddings that are not related to the image
    image_mask = processor.get_image_mask(batch_images)

    # Generate the similarity maps
    batched_similarity_maps = get_similarity_maps_from_embeddings(
        image_embeddings=image_embeddings,
        query_embeddings=query_embeddings,
        n_patches=n_patches,
        image_mask=image_mask,
    )

    # Get the similarity map for our (only) input image
    # (query_length, n_patches_x, n_patches_y)
    similarity_maps = batched_similarity_maps[0]

    # print(f"Similarity map shape: (query_length, n_patches_x, n_patches_y) = {tuple(similarity_maps.shape)}")

    top_percent = 0.1
    num_tokens = similarity_maps.shape[0]

    if not num_tokens:
        raise ValueError("No tokens in query")

    top_sim_maps = []
    for idx in range(num_tokens):
        max_sim_score = similarity_maps[idx, :, :].max().item()
        top_sim_maps.append((idx, max_sim_score))

    top_sim_maps.sort(key=lambda x: x[1], reverse=True)
    # take top tokens
    top_token_ids = [
        idx for idx, score in top_sim_maps[0 : ceil(num_tokens * top_percent)]
    ]
    current_similarity_map = torch.mean(similarity_maps[top_token_ids], dim=0)

    fig, ax = plot_similarity_map(
        image=image,
        similarity_map=current_similarity_map,
        figsize=(8, 8),
        show_colorbar=False,
    )

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="jpeg")
    ret = Image.open(img_buf)
    fig.clear()
    return ret
