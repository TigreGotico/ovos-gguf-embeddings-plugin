# GGUFTextEmbeddingsPlugin

The `GGUFTextEmbeddingsPlugin` is a plugin for recognizing and managing text embeddings. 

It integrates with [ovos-chromadb-embeddings-plugin](https://github.com/TigreGotico/ovos-chromadb-embeddings-plugin) for storing and retrieving text embeddings.

This plugin leverages the `llama-cpp-python` library to generate text embeddings. 

GGUF models are used to keep 3rd party dependencies to a minimum and ensuring this solver is lightweight and suitable for low powered hardware

## Features

- **Text Embeddings Extraction**: Converts text into embeddings using the `llama_cpp` model.
- **Text Data Storage**: Stores and retrieves text embeddings using `ChromaEmbeddingsDB`.
- **Text Data Management**: Allows for adding, querying, and deleting text embeddings associated with documents.

## Suggested Models

for english, [all-MiniLM-L6-v2](https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF) is recommended (20 MB)

if you need multilingual embeddings, [paraphrase-multilingual-minilm-l12-v2](https://huggingface.co/krogoldAI/paraphrase-multilingual-MiniLM-L12-v2-Q4_K_M-GGUF) is recommended (120MB)

By default `paraphrase-multilingual-minilm-l12-v2.Q4_K_M.gguf` will be used and downloaded on launch if missing

## Usage

Here is a quick example of how to use the `GGUFTextEmbeddingsPlugin`:

```python
from ovos_gguf_embeddings import GGUFTextEmbeddingsStore
from ovos_chromadb_embeddings import ChromaEmbeddingsDB

db = ChromaEmbeddingsDB("./my_db")
gguf = GGUFTextEmbeddingsStore(db, model=f"all-MiniLM-L6-v2.Q4_K_M.gguf")
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
```


