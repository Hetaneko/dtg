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
import threading

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
def process2(
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
        prompt: list = Body("query", title='prompt'),
        tag_length: list = Body("none", title='tag_length'),
        aspect_ratio: list = Body("none", title='aspect_ratio'),
        ban_tags: list = Body("none", title='ban_tags'),
        seed: list = Body("none", title='seed')
    ):
        models.model_dir = pathlib.Path(__file__).parent / "models"
        if len(prompt) > 1:
            allresults = []
            def task1():
                result = process(
                    "1girl,sitting",
                    aspect_ratio=float("1.2"),
                    seed=1,
                    tag_length=TOTAL_TAG_LENGTH["LONG"],
                    ban_tags="",
                    format=DEFAULT_FORMAT,
                    temperature=1.35,
                )
                allresults.append(result)
            def task2():
                result = process2(
                    "1girl,sitting",
                    aspect_ratio=float("1.2"),
                    seed=1,
                    tag_length=TOTAL_TAG_LENGTH["LONG"],
                    ban_tags="",
                    format=DEFAULT_FORMAT,
                    temperature=1.35,
                )
                allresults.append("result1")
            def task3():
                result = process(
                    "1girl,sitting",
                    aspect_ratio=float("1.2"),
                    seed=1,
                    tag_length=TOTAL_TAG_LENGTH["LONG"],
                    ban_tags="",
                    format=DEFAULT_FORMAT,
                    temperature=1.35,
                )
                allresults.append("result1")
            def task4():
                result = process(
                    "1girl,sitting",
                    aspect_ratio=float("1.2"),
                    seed=1,
                    tag_length=TOTAL_TAG_LENGTH["LONG"],
                    ban_tags="",
                    format=DEFAULT_FORMAT,
                    temperature=1.35,
                )
                allresults.append("result1")
            def task5():
                result = process(
                    "1girl,sitting",
                    aspect_ratio=float("1.2"),
                    seed=1,
                    tag_length=TOTAL_TAG_LENGTH["LONG"],
                    ban_tags="",
                    format=DEFAULT_FORMAT,
                    temperature=1.35,
                )
                allresults.append("result1")
            def task6():
                result = process(
                    "1girl,sitting",
                    aspect_ratio=float("1.2"),
                    seed=1,
                    tag_length=TOTAL_TAG_LENGTH["LONG"],
                    ban_tags="",
                    format=DEFAULT_FORMAT,
                    temperature=1.35,
                )
                allresults.append("result1")
                
            t1 = threading.Thread(target=task1, name='t1')
            t2 = threading.Thread(target=task2, name='t2')
            # t3 = threading.Thread(target=task3, name='t3')
            # t4 = threading.Thread(target=task4, name='t4')
            # t5 = threading.Thread(target=task5, name='t5')
            # t6 = threading.Thread(target=task6, name='t6')
        
            t1.start()
            t2.start()
            # t3.start()
            # t4.start()
            # t5.start()
            # t6.start()
        
            t1.join()
            t2.join()
            # t3.join()
            # t4.join()
            # t5.join()
            # t6.join()
            return allresults
        else:
            prompt = prompt[0]
            tag_length = tag_length[0]
            seed = int(seed[0])
            aspect_ratio = float(aspect_ratio[0])
            ban_tags = ban_tags[0]
    
            result = process(
                prompt,
                aspect_ratio=aspect_ratio,
                seed=seed,
                tag_length=TOTAL_TAG_LENGTH[tag_length],
                ban_tags=ban_tags,
                format=DEFAULT_FORMAT,
                temperature=1.35,
            )
            return [result]
try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(dtg_api)
except:
    pass
