import click
from quebra_frases import sentence_tokenize, paragraph_tokenize
from ovos_gguf_embeddings import GGUFTextEmbeddingsStore


@click.group()
def cli():
    """CLI for interacting with the GGUF Text Embeddings Store."""
    pass


@click.command()
@click.option('--database', prompt='Path to the ChromaDB database',
              help='Path to the ChromaDB database where the embeddings are stored.')
@click.option('--model', default='paraphrase-multilingual-minilm-l12-v2',
              help='Model name or URL used for generating embeddings. Defaults to "paraphrase-multilingual-minilm-l12-v2".')
@click.option('--top-k', default=5, help='Number of top results to return. Defaults to 5.')
@click.argument('query')
def query_document(query, database, model, top_k):
    """
    Query the embeddings store to find similar documents to the given query.

    QUERY: The query string used to search for similar documents.

    DATABASE: Path to the ChromaDB database where the embeddings are stored. Can be a full path or a simple string.
              If a simple string is provided, it will be saved in the XDG cache directory (~/.cache/chromadb/{database}).

    MODEL: Name or URL of the model used for generating embeddings. (Defaults to 'paraphrase-multilingual-minilm-l12-v2')

    TOP-K: Number of top results to return. (Defaults to 5)
    """
    gguf = GGUFTextEmbeddingsStore(db=database, model=model)
    docs = gguf.query(query, top_k=top_k)
    click.echo(f"Query results for '{query}':")
    for doc in docs:
        click.echo(f"Document: {doc[0]}, Distance: {doc[1]}")


@click.command()
@click.argument('document')
@click.option('--database', prompt='Path to the ChromaDB database',
              help='ChromaDB database where the embeddings are stored.')
@click.option('--model', default='paraphrase-multilingual-minilm-l12-v2',
              help='Model name or URL used for generating embeddings. Defaults to "paraphrase-multilingual-minilm-l12-v2".')
@click.option('--from-file', is_flag=True, help='Indicates if the document argument is a file path.')
@click.option('--use-sentences', is_flag=True,
              help='Indicates if the document should be tokenized into sentences; otherwise, it is split into paragraphs.')
def add_document(document, database, model, from_file, use_sentences):
    """
    Add a document to the embeddings store.

    DOCUMENT: The document string or file path to be added to the store.

    FROM-FILE: Flag indicating whether the DOCUMENT argument is a file path. If set, the file is read and processed.

    USE-SENTENCES: Flag indicating whether to tokenize the document into sentences. If not set, the document is split into paragraphs.

    DATABASE: Path to the ChromaDB database where the embeddings are stored. Can be a full path or a simple string.
              If a simple string is provided, it will be saved in the XDG cache directory (~/.cache/chromadb/{database}).

    MODEL: Name or URL of the model used for generating embeddings. (Defaults to 'paraphrase-multilingual-minilm-l12-v2')
    """
    gguf = GGUFTextEmbeddingsStore(db=database, model=model)

    if from_file:
        with open(document, "r") as f:
            document = f.read()
        if use_sentences:
            for p in paragraph_tokenize(document):
                for s in sentence_tokenize(p):
                    gguf.add_document(s)
        else:
            for p in paragraph_tokenize(document):
                gguf.add_document(p)
    else:
        gguf.add_document(document)

    click.echo(f"Document added: '{document[:30]}...'")  # Show only the first 30 chars for brevity


@click.command()
@click.argument('document')
@click.option('--database', prompt='Path to the ChromaDB database',
              help='ChromaDB database where the embeddings are stored.')
@click.option('--model', default='paraphrase-multilingual-minilm-l12-v2',
              help='Model name or URL used for generating embeddings. Defaults to "paraphrase-multilingual-minilm-l12-v2".')
def delete_document(document, database, model):
    """
    Delete a document from the embeddings store.

    DOCUMENT: The document string to be deleted from the store.

    DATABASE: Path to the ChromaDB database where the embeddings are stored. Can be a full path or a simple string.
              If a simple string is provided, it will be saved in the XDG cache directory (~/.cache/chromadb/{database}).

    MODEL: Name or URL of the model used for generating embeddings. (Defaults to 'paraphrase-multilingual-minilm-l12-v2')
    """
    gguf = GGUFTextEmbeddingsStore(db=database, model=model)
    gguf.delete_document(document)
    click.echo(f"Document deleted: '{document[:30]}...'")  # Show only the first 30 chars for brevity


cli.add_command(query_document)
cli.add_command(add_document)
cli.add_command(delete_document)

if __name__ == '__main__':
    cli()
