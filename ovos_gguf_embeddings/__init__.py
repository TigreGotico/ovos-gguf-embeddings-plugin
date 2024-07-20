import llama_cpp  # pip install llama-cpp-python
import numpy as np
from ovos_chromadb_embeddings import ChromaEmbeddingsDB
from ovos_plugin_manager.templates.embeddings import TextEmbeddingsStore, EmbeddingsDB


class GGUFTextEmbeddingRecognizer(TextEmbeddingsStore):
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

    docs = gguf.query("does the fish purr like a cat?", top_k=2)
    print(docs)
    # [('a cat is a feline and likes to purr', 0.6548102001030748),
    # ('a fish is a creature that lives in water and swims', 0.5436657174406345)]
