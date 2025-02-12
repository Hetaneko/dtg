import pathlib
import re
import random
from time import time

import torch
from transformers import AutoTokenizer, logging

import kgen.models as models
import kgen.executor.tipo as tipo
from kgen.formatter import seperate_tags, apply_format
from kgen.generate import generate
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
DEFAULT_FORMAT = """
<|special|>, <|characters|>, <|copyrights|>, 
<|artist|>, 

<|general|>,

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
    tipo.BAN_TAGS = seperate_tags(ban_tags.split(","))
    
    models.load_model(
    "TIPO-500M-ft-F16.gguf",
    gguf=True,
    device="cuda",
    main_gpu=1,
    )

    generate(max_new_tokens=4)

    def task(tags, nl_prompt):
        meta, operations, general, nl_prompt = tipo.parse_tipo_request(
            seperate_tags(tags.split(",")),
            nl_prompt,
            tag_length_target=tag_length,
            generate_extra_nl_prompt=not nl_prompt,
        )
        meta["aspect_ratio"] = f"{aspect_ratio:.1f}"
        result, timing = tipo.tipo_runner(meta, operations, general, nl_prompt)
        formatted = re.sub(r"([()\[\]])", r"\\\1", apply_format(result, DEFAULT_FORMAT))
        return formatted
    
    formatted = task(prompt, "")
    return formatted

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
