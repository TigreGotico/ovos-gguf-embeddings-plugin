import os.path
from typing import Optional, Union

import llama_cpp
import numpy as np
import requests
from ovos_chromadb_embeddings import ChromaEmbeddingsDB
from ovos_config.locations import get_xdg_cache_save_path
from ovos_plugin_manager.templates.embeddings import TextEmbeddingsStore, EmbeddingsDB
from ovos_utils.log import LOG


class GGUFTextEmbeddingsStore(TextEmbeddingsStore):
    DEFAULT_MODELS = {
        "all-MiniLM-L6-v2": "https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.Q4_K_M.gguf",
        "all-MiniLM-L12-v2": "https://huggingface.co/leliuga/all-MiniLM-L12-v2-GGUF/resolve/main/all-MiniLM-L12-v2.Q4_K_M.gguf",
        "multi-qa-MiniLM-L6-cos-v1": "https://huggingface.co/Felladrin/gguf-multi-qa-MiniLM-L6-cos-v1/resolve/main/multi-qa-MiniLM-L6-cos-v1.Q4_K_M.gguf",
        "gist-all-minilm-l6-v2": "https://huggingface.co/afrideva/GIST-all-MiniLM-L6-v2-GGUF/resolve/main/gist-all-minilm-l6-v2.Q4_K_M.gguf",
        "paraphrase-multilingual-minilm-l12-v2": "https://huggingface.co/krogoldAI/paraphrase-multilingual-MiniLM-L12-v2-Q4_K_M-GGUF/resolve/main/paraphrase-multilingual-minilm-l12-v2.Q4_K_M.gguf",
        "e5-small-v2": "https://huggingface.co/ChristianAzinn/e5-small-v2-gguf/resolve/main/e5-small-v2.Q4_K_M.gguf",
        "gte-small": "https://huggingface.co/ChristianAzinn/gte-small-gguf/resolve/main/gte-small.Q4_K_M.gguf",
        "gte-base": "https://huggingface.co/ChristianAzinn/gte-base-gguf/resolve/main/gte-base.Q4_K_M.gguf",
        "gte-large": "https://huggingface.co/ChristianAzinn/gte-large-gguf/resolve/main/gte-large.Q4_K_M.gguf",
        "snowflake-arctic-embed-l": "https://huggingface.co/ChristianAzinn/snowflake-arctic-embed-l-gguf/resolve/main/snowflake-arctic-embed-l--Q4_K_M.GGUF",
        "snowflake-arctic-embed-m": "https://huggingface.co/ChristianAzinn/snowflake-arctic-embed-m-gguf/resolve/main/snowflake-arctic-embed-m--Q4_K_M.GGUF",
        "snowflake-arctic-embed-m.long": "https://huggingface.co/ChristianAzinn/snowflake-arctic-embed-m-long-gguf/resolve/main/snowflake-arctic-embed-m-long-Q4_K_M.GGUF",
        "snowflake-arctic-embed-s": "https://huggingface.co/ChristianAzinn/snowflake-arctic-embed-s-gguf/resolve/main/snowflake-arctic-embed-s--Q4_K_M.GGUF",
        "snowflake-arctic-embed-xs": "https://huggingface.co/ChristianAzinn/snowflake-arctic-embed-xs-gguf/resolve/main/snowflake-arctic-embed-xs--Q4_K_M.GGUF",
        "nomic-embed-text-v1.5": "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf",
        "uae-large-v1": "https://huggingface.co/ChristianAzinn/uae-large-v1-gguf/resolve/main/uae-large-v1.Q4_K_M.gguf",
        "labse": "https://huggingface.co/ChristianAzinn/labse-gguf/resolve/main/labse.Q4_K_M.gguf",
        "bge-large-en-v1.5": "https://huggingface.co/ChristianAzinn/bge-large-en-v1.5-gguf/resolve/main/bge-large-en-v1.5.Q4_K_M.gguf",
        "bge-base-en-v1.5": "https://huggingface.co/ChristianAzinn/bge-base-en-v1.5-gguf/resolve/main/bge-base-en-v1.5.Q4_K_M.gguf",
        "bge-small-en-v1.5": "https://huggingface.co/ChristianAzinn/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5.Q4_K_M.gguf",
        "gist-large-embedding-v0": "https://huggingface.co/ChristianAzinn/gist-large-embedding-v0-gguf/resolve/main/gist-large-embedding-v0.Q4_K_M.gguf",
        "gist-embedding-v0": "https://huggingface.co/ChristianAzinn/gist-embedding-v0-gguf/resolve/main/gist-embedding-v0.Q4_K_M.gguf",
        "gist-small-embedding-v0": "https://huggingface.co/ChristianAzinn/gist-small-embedding-v0-gguf/resolve/main/gist-small-embedding-v0.Q4_K_M.gguf",
        "mxbai-embed-large-v1": "https://huggingface.co/ChristianAzinn/mxbai-embed-large-v1-gguf/resolve/main/mxbai-embed-large-v1.Q4_K_M.gguf",
        "acge_text_embedding": "https://huggingface.co/ChristianAzinn/acge_text_embedding-gguf/resolve/main/acge_text_embedding-Q4_K_M.GGUF",
        "gte-Qwen2-1.5B-instruct": "https://huggingface.co/second-state/gte-Qwen2-1.5B-instruct-GGUF/resolve/main/gte-Qwen2-1.5B-instruct-Q4_K_M.gguf",
        "gte-Qwen2-7B-instruct": "https://huggingface.co/niancheng/gte-Qwen2-7B-instruct-Q4_K_M-GGUF/resolve/main/gte-qwen2-7b-instruct-q4_k_m.gguf",
    }

    def __init__(self, db: Optional[Union[EmbeddingsDB, str]] = None,
                 model: str = "paraphrase-multilingual-minilm-l12-v2",
                 n_gpu_layers=0,
                 skip_db: bool=False):
        if model in self.DEFAULT_MODELS:
            model = self.DEFAULT_MODELS[model]
        if model.startswith("http"):
            model_path = f"{get_xdg_cache_save_path('gguf_models')}/{model.split('/')[-1]}"
            if not os.path.isfile(model_path):
                os.makedirs(get_xdg_cache_save_path("gguf_models"), exist_ok=True)
                LOG.info(f"Downloading {model}")
                data = requests.get(model).content
                with open(model_path, "wb") as f:
                    f.write(data)
            model = model_path

        if not skip_db:
            if db is None:
                db_path = get_xdg_cache_save_path("chromadb")
                os.makedirs(db_path, exist_ok=True)
                db = f"{db_path}/{model.split('/')[-1].split('.')[0]}"

            if isinstance(db, str):
                if "/" not in db:  # use xdg path
                    db = f"{get_xdg_cache_save_path('chromadb')}/{db}"
                LOG.info(f"Using chromadb as text embeddings store: {db}")
                db = ChromaEmbeddingsDB(db)

            super().__init__(db)

        LOG.info(f"Loading embeddings model: {model}")
        self.model = llama_cpp.Llama(
            model_path=model,
            verbose=False,
            n_gpu_layers=n_gpu_layers,
            embedding=True)

    def get_text_embeddings(self, text: str) -> np.ndarray:
        embeddings = self.model.create_embedding(text)
        e = embeddings["data"][0]['embedding']
        return e


if __name__ == "__main__":
    from os.path import dirname

    for m in GGUFTextEmbeddingsStore.DEFAULT_MODELS:
        print("Testing model:", m)
        gguf = GGUFTextEmbeddingsStore(model=m)
        corpus = [
            "a cat is a feline and likes to purr",
            "a dog is the human's best friend and loves to play",
            "a bird is a beautiful animal that can fly",
            "a fish is a creature that lives in water and swims",
        ]
        for s in corpus:
            gguf.add_document(s)

        docs = gguf.query("does the fish purr like a cat?", top_k=2)
        print(docs)

    # Testing model: all-MiniLM-L6-v2
    # 2024-07-21 01:31:46.362 - OVOS - __main__:__init__:42 - INFO - Using chromadb as text embeddings store: /home/miro/.cache/chromadb/all-MiniLM-L6-v2
    # 2024-07-21 01:31:46.442 - OVOS - __main__:__init__:47 - INFO - Loading embeddings model: /home/miro/.cache/gguf_models/all-MiniLM-L6-v2.Q4_K_M.gguf
    # [('a cat is a feline and likes to purr', 0.3451897998969252), ('a fish is a creature that lives in water and swims', 0.4563342825593655)]
    # Testing model: all-MiniLM-L12-v2
    # 2024-07-21 01:31:46.557 - OVOS - __main__:__init__:30 - INFO - Downloading https://huggingface.co/leliuga/all-MiniLM-L12-v2-GGUF/resolve/main/all-MiniLM-L12-v2.Q4_K_M.gguf
    # 2024-07-21 01:31:49.140 - OVOS - __main__:__init__:42 - INFO - Using chromadb as text embeddings store: /home/miro/.cache/chromadb/all-MiniLM-L12-v2
    # 2024-07-21 01:31:49.200 - OVOS - __main__:__init__:47 - INFO - Loading embeddings model: /home/miro/.cache/gguf_models/all-MiniLM-L12-v2.Q4_K_M.gguf
    # [('a cat is a feline and likes to purr', 0.2906984556278187), ('a fish is a creature that lives in water and swims', 0.5051804110638556)]
    # Testing model: multi-qa-MiniLM-L6-cos-v1
    # 2024-07-21 01:31:49.257 - OVOS - __main__:__init__:42 - INFO - Using chromadb as text embeddings store: /home/miro/.cache/chromadb/multi-qa-MiniLM-L6-cos-v1
    # 2024-07-21 01:31:49.261 - OVOS - __main__:__init__:47 - INFO - Loading embeddings model: /home/miro/.cache/gguf_models/multi-qa-MiniLM-L6-cos-v1.Q4_K_M.gguf
    # [('a cat is a feline and likes to purr', 0.30274518825428487), ('a fish is a creature that lives in water and swims', 0.5406810818023648)]
    # Testing model: gist-all-minilm-l6-v2
    # 2024-07-21 01:31:49.306 - OVOS - __main__:__init__:42 - INFO - Using chromadb as text embeddings store: /home/miro/.cache/chromadb/gist-all-minilm-l6-v2
    # 2024-07-21 01:31:49.310 - OVOS - __main__:__init__:47 - INFO - Loading embeddings model: /home/miro/.cache/gguf_models/gist-all-minilm-l6-v2.Q4_K_M.gguf
    # [('a cat is a feline and likes to purr', 0.11505124693827362), ('a fish is a creature that lives in water and swims', 0.17233980976734598)]
    # Testing model: paraphrase-multilingual-minilm-l12-v2
    # 2024-07-21 01:31:49.358 - OVOS - __main__:__init__:42 - INFO - Using chromadb as text embeddings store: /home/miro/.cache/chromadb/paraphrase-multilingual-minilm-l12-v2
    # 2024-07-21 01:31:49.362 - OVOS - __main__:__init__:47 - INFO - Loading embeddings model: /home/miro/.cache/gguf_models/paraphrase-multilingual-minilm-l12-v2.Q4_K_M.gguf
    # [('a fish is a creature that lives in water and swims', 0.16902063816774904), ('a cat is a feline and likes to purr', 0.2679446374144535)]
