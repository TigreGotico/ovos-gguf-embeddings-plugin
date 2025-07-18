import os
from typing import Dict, Any, Union
import numpy as np
import requests
from ovos_config.locations import get_xdg_cache_save_path
from ovos_plugin_manager.templates.embeddings import EmbeddingsArray, TextEmbedder
from ovos_utils.log import LOG
import llama_cpp


class GGUFEmbeddings(TextEmbedder):
    """
    A TextEmbedder implementation that uses GGUF models via llama_cpp for generating text embeddings.
    Models are automatically downloaded and cached locally.
    """

    DEFAULT_MODELS = {
        "all-MiniLM-L6-v2": "https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.Q4_K_M.gguf",
        "all-MiniLM-L12-v2": "https://huggingface.co/leliuga/all-MiniLM-L12-v2-GGUF/resolve/main/all-MiniLM-L12-v2.Q4_K_M.gguf",
        "multi-qa-MiniLM-L6-cos-v1": "https://huggingface.co/Felladrin/gguf-multi-qa-MiniLM-L6-cos-v1/resolve/main/multi-qa-MiniLM-L6-cos-v1.Q4_K_M.gguf",
        "gist-all-minilm-l6-v2": "https://huggingface.co/afrideva/GIST-all-MiniLM-L6-v2-GGUF/resolve/main/gist-all-minilm-l6-v2.Q4_K_M.gguf",
       # "paraphrase-multilingual-minilm-l12-v2": "https://huggingface.co/krogoldAI/paraphrase-multilingual-MiniLM-L12-v2-Q4_K_M-GGUF/resolve/main/paraphrase-multilingual-minilm-l12-v2.Q4_K_M.gguf",
        "e5-small-v2": "https://huggingface.co/ChristianAzinn/e5-small-v2-gguf/resolve/main/e5-small-v2.Q4_K_M.gguf",
        "gte-small": "https://huggingface.co/ChristianAzinn/gte-small-gguf/resolve/main/gte-small.Q4_K_M.gguf",
        "gte-base": "https://huggingface.co/ChristianAzinn/gte-base-gguf/resolve/main/gte-base.Q4_K_M.gguf",
        "gte-large": "https://huggingface.co/ChristianAzinn/gte-large-gguf/resolve/main/gte-large.Q4_K_M.gguf",
        "snowflake-arctic-embed-l": "https://huggingface.co/ChristianAzinn/snowflake-arctic-embed-l-gguf/resolve/main/snowflake-arctic-embed-l--Q4_K_M.GGUF",
        "snowflake-arctic-embed-m": "https://huggingface.co/ChristianAzinn/snowflake-arctic-embed-m-gguf/resolve/main/snowflake-arctic-embed-m--Q4_K_M.GGUF",
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

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize a GGUFEmbeddings instance with the specified configuration.
        
        Loads a GGUF model for text embedding generation based on the provided configuration dictionary. The model can be selected by name (from DEFAULT_MODELS), direct URL, or local file path. Additional configuration options are passed to the underlying llama_cpp.Llama model loader.
        """
        super().__init__(config)
        self.model = None
        self._load_model()

    @staticmethod
    def _download_model(url: str, model_path: str):
        """
        Download a GGUF model file from a specified URL to a local path, streaming the content and logging progress.
        
        Raises:
            requests.exceptions.RequestException: If the download fails due to a network or HTTP error.
            IOError: If writing the file to disk fails.
        """
        LOG.info(f"Downloading model from {url} to {model_path}")
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            response = requests.get(url, stream=True, timeout=300) # 5 minute timeout
            response.raise_for_status()  # Raise an exception for HTTP errors

            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192  # 8KB
            downloaded_size = 0

            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            LOG.debug(f"Download progress: {progress:.2f}% ({downloaded_size}/{total_size} bytes)")
                        else:
                            LOG.debug(f"Downloaded {downloaded_size} bytes...")
            LOG.info(f"Successfully downloaded model to {model_path}")
        except requests.exceptions.RequestException as e:
            LOG.error(f"Failed to download model from {url}: {e}")
            if os.path.exists(model_path):
                os.remove(model_path) # Clean up partial download
            raise
        except IOError as e:
            LOG.error(f"Failed to write model to {model_path}: {e}")
            raise

    def _load_model(self):
        """
        Load the GGUF embedding model according to the configuration, handling model selection, downloading, and initialization.
        
        Determines the model source from the configuration (default model name, URL, or local file path). Downloads the model if not already cached, and loads it using `llama_cpp.Llama` with appropriate arguments. If the model cannot be loaded, sets `self.model` to `None` and logs the error.
        """
        model_id = self.config.get("model", "labse")
        model_path: Union[str, None] = None

        if model_id in self.DEFAULT_MODELS:
            model_url = self.DEFAULT_MODELS[model_id]
            model_filename = model_url.split('/')[-1]
            model_path = os.path.join(get_xdg_cache_save_path('gguf_models'), model_filename)

            if not os.path.isfile(model_path):
                try:
                    self._download_model(model_url, model_path)
                except Exception as e:
                    LOG.error(f"Could not download and load model {model_id}: {e}")
                    self.model = None
                    return
            else:
                LOG.info(f"Using cached model: {model_path}")
        elif model_id.startswith("http"):
            model_url = model_id
            model_filename = model_url.split('/')[-1]
            model_path = os.path.join(get_xdg_cache_save_path('gguf_models'), model_filename)

            if not os.path.isfile(model_path):
                try:
                    self._download_model(model_url, model_path)
                except Exception as e:
                    LOG.error(f"Could not download and load model from URL {model_url}: {e}")
                    self.model = None
                    return
            else:
                LOG.info(f"Using cached model from URL: {model_path}")
        elif os.path.isfile(model_id):
            model_path = model_id
            LOG.info(f"Using local model file: {model_path}")
        else:
            LOG.error(f"Invalid model specified: {model_id}. Must be a default model name, URL, or local path.")
            self.model = None
            return

        if model_path:
            try:
                # Extract llama_cpp specific arguments from config
                llama_args = {k: v for k, v in self.config.items() if k not in ["model"]}
                llama_args.setdefault("n_gpu_layers", 0)
                llama_args.setdefault("verbose", False)
                llama_args.setdefault("embedding", True)

                LOG.info(f"Loading embeddings model: {model_path} with args: {llama_args}")
                self.model = llama_cpp.Llama(
                    model_path=model_path,
                    **llama_args
                )
            except Exception as e:
                LOG.error(f"Failed to load llama_cpp model from {model_path}: {e}")
                self.model = None
        else:
            raise ValueError(f"Invalid model_id '{model_id}'")

    def get_embeddings(self, text: str) -> EmbeddingsArray:
        """
        Generate embeddings for the given text using the loaded GGUF model.
        
        Parameters:
            text (str): Input text to generate embeddings for.
        
        Returns:
            EmbeddingsArray: NumPy array containing the embedding vector.
        
        Raises:
            RuntimeError: If the embedding model is not loaded.
            Exception: If embedding generation fails.
        """
        if self.model is None:
            raise RuntimeError("Embedding model not loaded. Please check logs for errors during initialization.")
        try:
            embeddings = self.model.create_embedding(text)
            e: np.ndarray = np.array(embeddings["data"][0]['embedding'])
            return e
        except Exception as e:
            LOG.error(f"Error generating embeddings for text '{text}': {e}")
            raise

if __name__ == "__main__":
    # Example usage:
    # Ensure you have llama-cpp-python installed: pip install llama-cpp-python
    # And ovos-plugin-manager: pip install ovos-plugin-manager ovos-utils ovos-config requests numpy

    # Test with a default model (will download if not cached)
    print("\n--- Testing default model (labse) ---")
    try:
        gguf_default = GGUFEmbeddings()
        corpus_default = [
            "The quick brown fox jumps over the lazy dog.",
            "A lazy dog is sleeping under a tree.",
            "Cats are known for their agility and grace.",
            "Dogs are loyal companions and often playful.",
        ]
        for s in corpus_default:
            try:
                embeddings = gguf_default.get_embeddings(s)
                print(f"Text: '{s}' | Embeddings shape: {embeddings.shape}")
            except Exception as e:
                print(f"Error getting embeddings for '{s}': {e}")
    except Exception as e:
        print(f"Failed to initialize GGUFEmbeddings with default model: {e}")

    # Test with a specific model from DEFAULT_MODELS
    print("\n--- Testing specific model (all-MiniLM-L6-v2) ---")
    try:
        gguf_minilm = GGUFEmbeddings(config={"model": "all-MiniLM-L6-v2", "n_gpu_layers": 0})
        corpus_minilm = [
            "This is a sentence about natural language processing.",
            "NLP is a field of artificial intelligence.",
        ]
        for s in corpus_minilm:
            try:
                embeddings = gguf_minilm.get_embeddings(s)
                print(f"Text: '{s}' | Embeddings shape: {embeddings.shape}")
            except Exception as e:
                print(f"Error getting embeddings for '{s}': {e}")
    except Exception as e:
        print(f"Failed to initialize GGUFEmbeddings with all-MiniLM-L6-v2: {e}")

    # Test with a non-existent model to demonstrate error handling
    print("\n--- Testing non-existent model (should fail) ---")
    try:
        gguf_invalid = GGUFEmbeddings(config={"model": "non-existent-model"})
        if gguf_invalid.model is None:
            print("Correctly failed to load non-existent model.")
        else:
            print("Unexpected: Non-existent model loaded successfully.")
    except Exception as e:
        print(f"Caught expected error for non-existent model: {e}")

    # Test with a local file path (replace with a real path if you have one)
    # print("\n--- Testing local file path (replace with your model path) ---")
    # local_model_path = "/path/to/your/model.gguf" # <<< CHANGE THIS
    # if os.path.exists(local_model_path):
    #     try:
    #         gguf_local = GGUFEmbeddings(config={"model": local_model_path})
    #         embeddings = gguf_local.get_embeddings("Hello world from local model!")
    #         print(f"Local model embeddings shape: {embeddings.shape}")
    #     except Exception as e:
    #         print(f"Failed to initialize GGUFEmbeddings with local model: {e}")
    # else:
    #     print(f"Skipping local model test: {local_model_path} does not exist.")

