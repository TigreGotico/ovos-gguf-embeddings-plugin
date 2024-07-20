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
        "paraphrase-multilingual-minilm-l12-v2": "https://huggingface.co/krogoldAI/paraphrase-multilingual-MiniLM-L12-v2-Q4_K_M-GGUF/resolve/main/paraphrase-multilingual-minilm-l12-v2.Q4_K_M.gguf"
    }

    def __init__(self, db: Optional[Union[EmbeddingsDB, str]] = None,
                 model: str = "paraphrase-multilingual-minilm-l12-v2"):
        if model in self.DEFAULT_MODELS:
            model = self.DEFAULT_MODELS[model]
        if model.startswith("http"):
            model_path = f"{get_xdg_cache_save_path("gguf_models")}/{model.split('/')[-1]}"
            if not os.path.isfile(model_path):
                os.makedirs(get_xdg_cache_save_path("gguf_models"), exist_ok=True)
                LOG.info(f"Downloading {model}")
                data = requests.get(model).content
                with open(model_path, "wb") as f:
                    f.write(data)
            model = model_path

        if db is None:
            db_path = get_xdg_cache_save_path("chromadb")
            os.makedirs(db_path, exist_ok=True)
            db = f"{db_path}/{model.split('/')[-1].split('.')[0]}"

        if isinstance(db, str):
            LOG.info(f"Using chromadb as text embeddings store: {db}")
            db = ChromaEmbeddingsDB(db)

        super().__init__(db)

        LOG.info(f"Loading embeddings model: {model}")
        self.model = llama_cpp.Llama(
            model_path=model,
            verbose=False,
            embedding=True)

    def get_text_embeddings(self, text: str) -> np.ndarray:
        embeddings = self.model.create_embedding(text)
        return embeddings["data"][0]['embedding']


if __name__ == "__main__":
    from os.path import dirname

    gguf = GGUFTextEmbeddingsStore()
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
    # [('a cat is a feline and likes to purr', 0.6548102001030748),
    # ('a fish is a creature that lives in water and swims', 0.5436657174406345)]
