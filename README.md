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

| Model Name                            | URL                                                                                                                                                       | Description                                                                                                                                                                                                                                                                                                                 | Suggested Use Cases                                                                                                                                                               |
|---------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| all-MiniLM-L6-v2                      | [Link](https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.Q4_K_M.gguf)                                                    | A sentence-transformers model that maps sentences & paragraphs to a 384-dimensional dense vector space. Fine-tuned on a 1B sentence pairs dataset using contrastive learning. Ideal for general-purpose tasks like information retrieval, clustering, and sentence similarity.                                              | Suitable for tasks that require fast inference and can handle slightly less accuracy, such as real-time applications.                                                             |
| all-MiniLM-L12-v2                     | [Link](https://huggingface.co/leliuga/all-MiniLM-L12-v2-GGUF/resolve/main/all-MiniLM-L12-v2.Q4_K_M.gguf)                                                  | A larger MiniLM model mapping sentences & paragraphs to a 384-dimensional dense vector space. Fine-tuned on a 1B sentence pairs dataset using contrastive learning. Provides higher accuracy for complex tasks.                                                                                                             | Suitable for more complex NLP tasks requiring higher accuracy, such as detailed semantic analysis, document ranking, and clustering.                                              |
| multi-qa-MiniLM-L6-cos-v1             | [Link](https://huggingface.co/Felladrin/gguf-multi-qa-MiniLM-L6-cos-v1/resolve/main/multi-qa-MiniLM-L6-cos-v1.Q4_K_M.gguf)                                | A sentence-transformers model mapping sentences & paragraphs to a 384-dimensional dense vector space, trained on 215M QA pairs. Designed for semantic search.                                                                                                                                                               | Best for semantic search, encoding queries/questions, and finding relevant documents or passages in QA tasks.                                                                     |
| gist-all-minilm-l6-v2                 | [Link](https://huggingface.co/afrideva/GIST-all-MiniLM-L6-v2-GGUF/resolve/main/gist-all-minilm-l6-v2.Q4_K_M.gguf)                                         | Enhanced version of all-MiniLM-L6-v2 using GISTEmbed method, improving in-batch negative selection during training. Demonstrates state-of-the-art performance on specific tasks with a focus on reducing data noise and improving model fine-tuning.                                                                        | Ideal for high-accuracy retrieval tasks, semantic search, and applications requiring efficient smaller models with robust performance, such as resource-constrained environments. |
| paraphrase-multilingual-minilm-l12-v2 | [Link](https://huggingface.co/krogoldAI/paraphrase-multilingual-MiniLM-L12-v2-Q4_K_M-GGUF/resolve/main/paraphrase-multilingual-minilm-l12-v2.Q4_K_M.gguf) | A sentence-transformers model mapping sentences & paragraphs to a 384-dimensional dense vector space. Supports multiple languages, optimized for paraphrasing tasks.                                                                                                                                                        | Perfect for multilingual applications, translation services, and tasks requiring paraphrase detection and generation.                                                             |
| e5-small-v2                           | [Link](https://huggingface.co/ChristianAzinn/e5-small-v2-gguf/resolve/main/e5-small-v2.Q4_K_M.gguf)                                                       | Text Embeddings by Weakly-Supervised Contrastive Pre-training. This model has 12 layers and the embedding size is 384. Size is about 30MB.                                                                                                                                                                                  | Ideal for applications requiring efficient, small-sized models with robust text embeddings.                                                                                       |
| gte-small                             | [Link](https://huggingface.co/ChristianAzinn/gte-small-gguf/resolve/main/gte-small.Q4_K_M.gguf)                                                           | General Text Embeddings (GTE) model. Trained using multi-stage contrastive learning by Alibaba DAMO Academy. Based on the BERT framework, it covers a wide range of domains and scenarios. About 30MB.                                                                                                                      | Suitable for information retrieval, semantic textual similarity, text reranking, and various other downstream tasks requiring text embeddings.                                    |
| gte-base                              | [Link](https://huggingface.co/ChristianAzinn/gte-base-gguf/resolve/main/gte-base.Q4_K_M.gguf)                                                             | Larger version of previous model, about 75 MB                                                                                                                                                                                                                                                                               |                                                                                                                                                                                   |
| gte-large                             | [Link](https://huggingface.co/ChristianAzinn/gte-large-gguf/resolve/main/gte-large.Q4_K_M.gguf)                                                           | Larger version of previous model, about 220 MB                                                                                                                                                                                                                                                                              |                                                                                                                                                                                   |
| snowflake-arctic-embed-l              | [Link](https://huggingface.co/ChristianAzinn/snowflake-arctic-embed-l-gguf/resolve/main/snowflake-arctic-embed-l--Q4_K_M.GGUF)                            | Part of the snowflake-arctic-embed suite, this model focuses on high-quality retrieval and achieves state-of-the-art performance on the MTEB/BEIR leaderboard. Trained using a multi-stage pipeline with a mix of public and proprietary data. About 215MB.                                                                 | Optimized for high-performance text retrieval tasks and achieving top accuracy in retrieval benchmarks.                                                                           |
| snowflake-arctic-embed-m              | [Link](https://huggingface.co/ChristianAzinn/snowflake-arctic-embed-m-gguf/resolve/main/snowflake-arctic-embed-m--Q4_K_M.GGUF)                            | Based on the intfloat/e5-base-unsupervised model, this medium-sized model balances high retrieval performance with efficient inference. About 75MB                                                                                                                                                                          | Ideal for general-purpose retrieval tasks requiring a balance between performance and efficiency.                                                                                 |
| snowflake-arctic-embed-m.long         | [Link](https://huggingface.co/ChristianAzinn/snowflake-arctic-embed-m-long-gguf/resolve/main/snowflake-arctic-embed-m-long-Q4_K_M.GGUF)                   | Based on the nomic-ai/nomic-embed-text-v1-unsupervised model, this long-context variant supports up to 2048 tokens without RPE and up to 8192 tokens with RPE. Perfect for long-context workloads. About 90MB                                                                                                               | Suitable for tasks requiring long-context embeddings, such as complex document analysis or extensive information retrieval.                                                       |
| snowflake-arctic-embed-s              | [Link](https://huggingface.co/ChristianAzinn/snowflake-arctic-embed-s-gguf/resolve/main/snowflake-arctic-embed-s--Q4_K_M.GGUF)                            | 	Based on the intfloat/e5-small-unsupervised model, this small model offers high retrieval accuracy with only 33M parameters and 384 dimensions.                                                                                                                                                                            | Suitable for applications needing efficient, high-accuracy retrieval in constrained environments.                                                                                 |
| snowflake-arctic-embed-xs             | [Link](https://huggingface.co/ChristianAzinn/snowflake-arctic-embed-xs-gguf/resolve/main/snowflake-arctic-embed-xs--Q4_K_M.GGUF)                          | 	Based on the all-MiniLM-L6-v2 model, this tiny model has only 22M parameters and 384 dimensions, providing a balance of low latency and high retrieval accuracy.                                                                                                                                                           | Best for ultra-low latency applications with strict size and cost constraints.                                                                                                    |
| nomic-embed-text-v1.5                 | [Link](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf)                                         | About 85MB. Resizable Production Embeddings with Matryoshka Representation Learning. The model is trained in two stages, starting with unsupervised contrastive learning on weakly related text pairs, followed by finetuning with high-quality labeled datasets. It is now multimodal, aligning with nomic-embed-vision-v1 | Ideal for applications requiring flexible embedding sizes and multimodal capabilities.                                                                                            |
| uae-large-v1                          | [Link](https://huggingface.co/ChristianAzinn/uae-large-v1-gguf/resolve/main/uae-large-v1.Q4_K_M.gguf)                                                     | Universal AnglE Embedding. AnglE-optimized Text Embeddings with a novel angle optimization approach.  About 220MB.                                                                                                                                                                                                          | Best for high-quality text embeddings in semantic textual similarity tasks, including short-text and long-text STS.                                                               |
| labse                                 | [Link](https://huggingface.co/ChristianAzinn/labse-gguf/resolve/main/labse.Q4_K_M.gguf)                                                                   | 	A port of the LaBSE model. Maps 109 languages to a shared vector space, supports up to 512 tokens of context. The model is optimized for producing similar representations for bilingual sentence pairs. About 390MB.                                                                                                      | Suitable for multilingual applications, translation mining, and cross-lingual text embedding tasks.                                                                               |
| bge-large-en-v1.5                     | [Link](https://huggingface.co/ChristianAzinn/bge-large-en-v1.5-gguf/resolve/main/bge-large-en-v1.5.Q4_K_M.gguf)                                           | The model is part of the BGE series and is designed for diverse retrieval tasks. Size is 216MB.                                                                                                                                                                                                                             |                                                                                                                                                                                   |
| bge-base-en-v1.5                      | [Link](https://huggingface.co/ChristianAzinn/bge-base-en-v1.5-gguf/resolve/main/bge-base-en-v1.5.Q4_K_M.gguf)                                             | Medium version of the above. About 80MB                                                                                                                                                                                                                                                                                     |                                                                                                                                                                                   |
| bge-small-en-v1.5                     | [Link](https://huggingface.co/ChristianAzinn/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5.Q4_K_M.gguf)                                           | Small version of the above. About 30MB                                                                                                                                                                                                                                                                                      |                                                                                                                                                                                   |
| gist-embedding-v0                     | [Link](https://huggingface.co/ChristianAzinn/gist-embedding-v0-gguf/resolve/main/gist-embedding-v0.Q4_K_M.gguf)                                           | GISTEmbed: Guided In-sample Selection of Training Negatives for Text Embedding Fine-tuning. Fine-tuned on top of the BAAI/bge-base-en-v1.5 using the MEDI dataset augmented with mined triplets from the MTEB Classification training dataset.                                                                              | Ideal for applications requiring embeddings without crafting instructions for queries.                                                                                            |
| gist-large-embedding-v0               | [Link](https://huggingface.co/ChristianAzinn/gist-large-embedding-v0-gguf/resolve/main/gist-large-embedding-v0.Q4_K_M.gguf)                               | Large version of the model above                                                                                                                                                                                                                                                                                            |                                                                                                                                                                                   |
| gist-small-embedding-v0               | [Link](https://huggingface.co/ChristianAzinn/gist-small-embedding-v0-gguf/resolve/main/gist-small-embedding-v0.Q4_K_M.gguf)                               | Small version of the model above                                                                                                                                                                                                                                                                                            |                                                                                                                                                                                   |
| mxbai-embed-large-v1                  | [Link](https://huggingface.co/ChristianAzinn/mxbai-embed-large-v1-gguf/resolve/main/mxbai-embed-large-v1.Q4_K_M.gguf)                                     | trained using AnglE loss on our high-quality large scale data. It achieves SOTA performance on BERT-large scale. About 220MB.                                                                                                                                                                                               | Best for tasks requiring high precision and detailed embeddings. Provides state-of-the-art performance among efficiently sized models.                                            |
| acge_text_embedding                   | [Link](https://huggingface.co/ChristianAzinn/acge_text_embedding-gguf/resolve/main/acge_text_embedding-Q4_K_M.GGUF)                                       | The ACGE model is developed by the Huhu Information Technology team on the TextIn platform. It is a general-purpose text encoding model that uses Matryoshka Representation Learning for variable-length vectorization. About 200MB                                                                                         | Ideal for chinese text                                                                                                                                                            |
| gte-Qwen2-7B-instruct                 | [Link](https://huggingface.co/niancheng/gte-Qwen2-7B-instruct-Q4_K_M-GGUF/resolve/main/gte-qwen2-7b-instruct-q4_k_m.gguf)                                 | 	The latest in the GTE model family, ranking No.1 in English and Chinese evaluations on the MTEB benchmark. Based on the Qwen2-7B LLM model, it integrates bidirectional attention mechanisms and instruction tuning, with comprehensive multilingual training. 4.68GB                                                      | Best for high-performance multilingual text embeddings and complex tasks requiring top-tier contextual understanding.                                                             |
| gte-Qwen2-1.5B-instruct               | [Link](https://huggingface.co/second-state/gte-Qwen2-1.5B-instruct-GGUF/resolve/main/gte-Qwen2-1.5B-instruct-Q4_K_M.gguf)                                 | gte-Qwen2-1.5B-instruct is the latest model in the gte (General Text Embedding) model family. The model is built on Qwen2-1.5B LLM model and use the same training data and strategies as the gte-Qwen2-7B-instruct model. 1.12GB                                                                                           |                                                                                                                                                                                   |

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

