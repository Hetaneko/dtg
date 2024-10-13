import time
import pathlib

import kgen.models as models
from kgen.formatter import seperate_tags, apply_format
from kgen.metainfo import TARGET
from kgen.executor.dtg import tag_gen, apply_dtg_prompt
from kgen.logging import logger

import numpy as np
from fastapi import FastAPI, Body
from fastapi.exceptions import HTTPException
import requests
import os
import gradio as gr
import subprocess
import modules.shared

from modules.api.models import *
from modules.api import api

SEED_MAX = 2**31 - 1
TOTAL_TAG_LENGTH = {
    "VERY_SHORT": "very short",
    "SHORT": "short",
    "LONG": "long",
    "VERY_LONG": "very long",
}
DEFAULT_FORMAT = """<|special|>, <|characters|>, <|copyrights|>, 
<|artist|>, 

<|extended|>.

<|general|>,

<|generated|>.

<|quality|>, <|meta|>, <|rating|>
"""


def process(
    prompt: str,
    aspect_ratio: float,
    seed: int,
    tag_length: str,
    ban_tags: str,
    format: str,
    temperature: float,
):
    propmt_preview = prompt.replace("\n", " ")[:40]
    logger.info(f"Processing propmt: {propmt_preview}...")
    logger.info(f"Processing with seed: {seed}")
    black_list = [tag.strip() for tag in ban_tags.split(",") if tag.strip()]
    all_tags = [tag.strip().lower() for tag in prompt.strip().split(",") if tag.strip()]

    tag_length = tag_length.replace(" ", "_")
    len_target = TARGET[tag_length]

    tag_map = seperate_tags(all_tags)
    dtg_prompt = apply_dtg_prompt(tag_map, tag_length, aspect_ratio)
    for _, extra_tokens, iter_count in tag_gen(
        models.text_model,
        models.tokenizer,
        dtg_prompt,
        tag_map["special"] + tag_map["general"],
        len_target,
        black_list,
        temperature=temperature,
        top_p=0.95,
        top_k=100,
        max_new_tokens=256,
        max_retry=20,
        max_same_output=15,
        seed=seed % SEED_MAX,
    ):
        pass
    tag_map["general"] += extra_tokens
    prompt_by_dtg = apply_format(tag_map, format)
    logger.info(
        "Prompt processing done. General Tags Count: "
        f"{len(tag_map['general'] + tag_map['special'])}"
        f" | Total iterations: {iter_count}"
    )
    return prompt_by_dtg

def dtg_api(_: gr.Blocks, app: FastAPI):
    @app.post("/dtg/process")
    async def dtg(
        prompt: str = Body("query", title='prompt'),
        tag_length: str = Body("none", title='tag_length'),
        aspect_ratio: str = Body("none", title='aspect_ratio'),
        ban_tags: str = Body("none", title='ban_tags'),
        seed: str = Body("none", title='seed')
    ):
        seed = int(seed)
        models.model_dir = pathlib.Path(__file__).parent / "models"
        aspect_ratio = float(aspect_ratio)
        # file = models.download_gguf(gguf_name="ggml-model-Q6_K.gguf")
        files = models.list_gguf()
        file = files[-1]
        logger.info(f"Use gguf model from local file: {file}")
        models.load_model(file, gguf=True, device="cpu")
        # models.load_model()
        # models.text_model.half().cuda()

        result = process(
            prompt,
            aspect_ratio=aspect_ratio,
            seed=seed,
            tag_length=TOTAL_TAG_LENGTH[tag_length],
            ban_tags=ban_tags,
            format=DEFAULT_FORMAT,
            temperature=1.35,
        )
        return result
try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(dtg_api)
except:
    pass
