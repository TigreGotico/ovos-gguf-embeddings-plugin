# GGUFTextEmbeddingsPlugin

The `GGUFTextEmbeddingsPlugin` is a plugin for recognizing and managing text embeddings.

It integrates with [ovos-chromadb-embeddings-plugin](https://github.com/TigreGotico/ovos-chromadb-embeddings-plugin) for
storing and retrieving text embeddings.

This plugin leverages the `llama-cpp-python` library to generate text embeddings.

GGUF models are used to keep 3rd party dependencies to a minimum and ensuring this solver is lightweight and suitable
for low powered hardware

## Features

- **Text Embeddings Extraction**: Converts text into embeddings using the `llama_cpp` model.
- **Text Data Storage**: Stores and retrieves text embeddings using `ChromaEmbeddingsDB`.
- **Text Data Management**: Allows for adding, querying, and deleting text embeddings associated with documents.

## Suggested Models

You can specify a downloaded model path, or use one of the pre-defined model strings in the table below.

If needed a model will be automatically downloaded to `~/.cache/gguf_models`

| Model Name                             | URL                                                                                                               | Description                                                                                                           | Suggested Use Cases                                                                                                    |
|----------------------------------------|-------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| all-MiniLM-L6-v2                       | [Link](https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.Q4_K_M.gguf)             | A sentence-transformers model that maps sentences & paragraphs to a 384-dimensional dense vector space. Fine-tuned on a 1B sentence pairs dataset using contrastive learning. Ideal for general-purpose tasks like information retrieval, clustering, and sentence similarity. | Suitable for tasks that require fast inference and can handle slightly less accuracy, such as real-time applications. |
| all-MiniLM-L12-v2                      | [Link](https://huggingface.co/leliuga/all-MiniLM-L12-v2-GGUF/resolve/main/all-MiniLM-L12-v2.Q4_K_M.gguf)           | A larger MiniLM model mapping sentences & paragraphs to a 384-dimensional dense vector space. Fine-tuned on a 1B sentence pairs dataset using contrastive learning. Provides higher accuracy for complex tasks. | Suitable for more complex NLP tasks requiring higher accuracy, such as detailed semantic analysis, document ranking, and clustering. |
| multi-qa-MiniLM-L6-cos-v1              | [Link](https://huggingface.co/Felladrin/gguf-multi-qa-MiniLM-L6-cos-v1/resolve/main/multi-qa-MiniLM-L6-cos-v1.Q4_K_M.gguf) | A sentence-transformers model mapping sentences & paragraphs to a 384-dimensional dense vector space, trained on 215M QA pairs. Designed for semantic search. | Best for semantic search, encoding queries/questions, and finding relevant documents or passages in QA tasks. |
| gist-all-minilm-l6-v2                  | [Link](https://huggingface.co/afrideva/GIST-all-MiniLM-L6-v2-GGUF/resolve/main/gist-all-minilm-l6-v2.Q4_K_M.gguf)  | Enhanced version of all-MiniLM-L6-v2 using GISTEmbed method, improving in-batch negative selection during training. Demonstrates state-of-the-art performance on specific tasks with a focus on reducing data noise and improving model fine-tuning. | Ideal for high-accuracy retrieval tasks, semantic search, and applications requiring efficient smaller models with robust performance, such as resource-constrained environments. |
| paraphrase-multilingual-minilm-l12-v2  | [Link](https://huggingface.co/krogoldAI/paraphrase-multilingual-MiniLM-L12-v2-Q4_K_M-GGUF/resolve/main/paraphrase-multilingual-minilm-l12-v2.Q4_K_M.gguf) | A sentence-transformers model mapping sentences & paragraphs to a 384-dimensional dense vector space. Supports multiple languages, optimized for paraphrasing tasks. | Perfect for multilingual applications, translation services, and tasks requiring paraphrase detection and generation. |


By default `paraphrase-multilingual-minilm-l12-v2` will be used if model is not specified

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


### CLI Interface

```bash
$ovos-gguf-embeddings --help 
Usage: ovos-gguf-embeddings [OPTIONS] COMMAND [ARGS]...

  CLI for interacting with the GGUF Text Embeddings Store.

Options:
  --help  Show this message and exit.

Commands:
  add-document     Add a document to the embeddings store.
  delete-document  Delete a document from the embeddings store.
  query-document   Query the embeddings store to find similar documents...
```

```bash
$ovos-gguf-embeddings add-document --help 
Usage: ovos-gguf-embeddings add-document [OPTIONS] DOCUMENT

  Add a document to the embeddings store.

  DOCUMENT: The document string or file path to be added to the store.

  FROM-FILE: Flag indicating whether the DOCUMENT argument is a file path. If
  set, the file is read and processed.

  USE-SENTENCES: Flag indicating whether to tokenize the document into
  sentences. If not set, the document is split into paragraphs.

  DATABASE: Path to the ChromaDB database where the embeddings are stored.
  (Required)

  MODEL: Name or URL of the model used for generating embeddings. (Defaults to
  'paraphrase-multilingual-minilm-l12-v2')

Options:
  --database TEXT  Path to the ChromaDB database where the embeddings are
                   stored.
  --model TEXT     Model name or URL used for generating embeddings. Defaults
                   to "paraphrase-multilingual-minilm-l12-v2".
  --from-file      Indicates if the document argument is a file path.
  --use-sentences  Indicates if the document should be tokenized into
                   sentences; otherwise, it is split into paragraphs.
  --help           Show this message and exit.
```

```bash
$ovos-gguf-embeddings query-document --help 
Usage: ovos-gguf-embeddings query-document [OPTIONS] QUERY

  Query the embeddings store to find similar documents to the given query.

  QUERY: The query string used to search for similar documents.

  DATABASE: Path to the ChromaDB database where the embeddings are stored. Can
  be a full path or a simple string.           If a simple string is provided,
  it will be saved in the XDG cache directory (~/.cache/chromadb/{database}).

  MODEL: Name or URL of the model used for generating embeddings. (Defaults to
  'paraphrase-multilingual-minilm-l12-v2')

  TOP-K: Number of top results to return. (Defaults to 5)

Options:
  --database TEXT  Path to the ChromaDB database where the embeddings are
                   stored.
  --model TEXT     Model name or URL used for generating embeddings. Defaults
                   to "paraphrase-multilingual-minilm-l12-v2".
  --top-k INTEGER  Number of top results to return. Defaults to 5.
  --help           Show this message and exit.

```

```bash
$ovos-gguf-embeddings delete-document --help 
Usage: ovos-gguf-embeddings delete-document [OPTIONS] DOCUMENT

  Delete a document from the embeddings store.

  DOCUMENT: The document string to be deleted from the store.

  DATABASE: Path to the ChromaDB database where the embeddings are stored. Can
  be a full path or a simple string.           If a simple string is provided,
  it will be saved in the XDG cache directory (~/.cache/chromadb/{database}).

  MODEL: Name or URL of the model used for generating embeddings. (Defaults to
  'paraphrase-multilingual-minilm-l12-v2')

Options:
  --database TEXT  ChromaDB database where the embeddings are stored.
  --model TEXT     Model name or URL used for generating embeddings. Defaults
                   to "paraphrase-multilingual-minilm-l12-v2".
  --help           Show this message and exit.
```

