import os
import torch
from google import genai
from google.genai import types

# ── Config ─────────────────────────────────────────────────────────────────────
API_KEY = "AIzaSyA9Cg1HxoqnBI0FRLmkWvL00iRdHg_AtFI"

DESCRIPTION_MODEL = "gemini-2.5-flash-lite"
EMBEDDING_MODEL   = "gemini-embedding-001"

# Single client instance shared by both functions
_client = genai.Client(api_key=API_KEY)

# ── Description ────────────────────────────────────────────────────────────────

def generate_species_description(species_name: str) -> str:
    """
    Generate a dense ecological description for a single species (or concept).

    Args:
        species_name: Scientific name or any ecological concept.

    Returns:
        Plain-text paragraph (~60-80 words) covering climate, soil,
        topography, and canopy structure. Fed directly into the text encoder.
    """
    prompt = f"""You are an expert ecologist writing training data for a Species \
Distribution Model (SDM). The SDM was trained exclusively on plant habitat profiles \
written in standard ecological terminology covering climate, soil, topography, and \
vegetation structure.

Your task: given any input — a species name, an ecosystem type, a geographic feature, \
a climate pattern, or any other concept — write a "Semantic Habitat Profile" that \
describes the terrestrial habitat most strongly associated with that concept, using \
ONLY the vocabulary and dimensions the SDM understands.

Input: {species_name}

Instructions:
- If the input is a plant species (or genus): describe its actual habitat directly.
- If the input is an animal, fungus, or other organism: describe the terrestrial \
habitat it occupies or depends on (e.g. for a salmon, describe the riparian forest \
and stream-bank habitat, not the fish itself).
- If the input is an ecosystem, biome, or geographic feature: describe the \
characteristic terrestrial habitat of that system.
- If the input is a climate pattern, disturbance type, or abiotic process: describe \
the habitat where that process dominates or has the strongest ecological signature.
- If the input is unclear or invalid: describe the most plausible habitat for the \
closest recognizable taxon or concept.

Write approximately 60-80 words as a single cohesive paragraph. No bullet points.

Focus ONLY on these four dimensions — do not mention the input concept by name \
after the first sentence:
1. Climate: temperature tolerance, seasonality, precipitation, moisture regime.
2. Soil: texture (sand/silt/clay/peat), acidity (pH), drainage, organic matter.
3. Topography: elevation range, slope position, aspect, landform.
4. Vegetation structure: canopy type (deciduous/coniferous/shrub/open), \
shade tolerance, successional stage.

Use precise scientific terms (e.g. "mesic upland", "calcareous till", \
"boreal mixed-wood", "podzolic soil", "subalpine krummholz").
If you genuinely lack information, incorporate "insufficient data" naturally \
into the paragraph rather than omitting dimensions.
Return ONLY the description paragraph. No JSON, no labels, no preamble."""

    try:
        response = _client.models.generate_content(
            model=DESCRIPTION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=300,
            ),
        )
        return response.text.strip()
    except Exception as e:
        print(f"Warning: description generation failed for '{species_name}': {e}")
        return (
            f"{species_name} is a plant species with specific habitat requirements "
            f"and climate preferences. Detailed description unavailable."
        )


# ── Embedding ──────────────────────────────────────────────────────────────────

def generate_species_embedding(description: str) -> torch.Tensor:
    """
    Encode a text description into a 768-dim embedding vector.

    Args:
        description: Output of generate_species_description()

    Returns:
        torch.Tensor of shape (768,), dtype float32, on CPU.
    """
    result = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=description,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            title="Species Habitat Profile",
            output_dimensionality=768,
        ),
    )
    # result.embeddings is a list of ContentEmbedding objects; we pass one string
    # so there is always exactly one item.
    vector = result.embeddings[0].values
    return torch.tensor(vector, dtype=torch.float32)