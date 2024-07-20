from typing import List, Tuple

import llama_cpp  # pip install llama-cpp-python
import numpy as np
from ovos_chromadb_embeddings import ChromaEmbeddingsDB
from ovos_plugin_manager.templates.embeddings import TextEmbeddingRecognizer, EmbeddingsDB


class GGUFTextEmbeddingRecognizer(TextEmbeddingRecognizer):
    def __init__(self, db: EmbeddingsDB, model: str = None):
        super().__init__(db)
        mdl = model or f"{dirname(__file__)}/all-MiniLM-L6-v2.Q4_K_M.gguf"
        self.model = llama_cpp.Llama(
            model_path=mdl,
            verbose=False,
            embedding=True)

    def get_text_embeddings(self, text: str) -> np.ndarray:
        embeddings = self.model.create_embedding(text)
        return embeddings["data"][0]['embedding']

    def add_document(self, document: str) -> None:
        embeddings = self.model.create_embedding(document)
        emb = embeddings["data"][0]['embedding']
        self.db.add_embeddings(document, emb)

    def delete_document(self, document: str) -> None:
        self.db.delete_embedding(document)

    def query_document(self, document, top_k: int = 5) -> List[Tuple[str, float]]:
        return [(doc, 1 - conf) for doc, conf in
                super().query_document(document, top_k)]


if __name__ == "__main__":
    from os.path import dirname

    db = ChromaEmbeddingsDB("/tmp/my_db")
    gguf = GGUFTextEmbeddingRecognizer(db, model=f"{dirname(__file__)}/all-MiniLM-L6-v2.Q4_K_M.gguf")
    corpus = [
        "a cat is a feline and likes to purr",
        "a dog is the human's best friend and loves to play",
        "a bird is a beautiful animal that can fly",
        "a fish is a creature that lives in water and swims",
    ]
    for s in corpus:
        gguf.add_document(s)

    docs = gguf.query_document("does the fish purr like a cat?", top_k=2)
    print(docs)
    # [('a cat is a feline and likes to purr', 0.6548102001030748),
    # ('a fish is a creature that lives in water and swims', 0.5436657174406345)]
