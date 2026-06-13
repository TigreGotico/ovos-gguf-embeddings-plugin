import unittest
from unittest.mock import MagicMock, patch
import numpy as np


def _make_fake_llama_cpp():
    fake_llama_cpp = MagicMock()
    fake_model = MagicMock()
    fake_model.create_embedding.return_value = {
        "data": [{"embedding": [0.1, 0.2, 0.3]}]
    }
    fake_llama_cpp.Llama.return_value = fake_model
    return fake_llama_cpp, fake_model


class TestGGUFEmbeddingsLocalFile(unittest.TestCase):
    """(a) local .gguf path that os.path.isfile says exists loads the model."""

    def test_local_file_loads_and_embeds(self):
        fake_llama_cpp, fake_model = _make_fake_llama_cpp()

        with patch("ovos_gguf_embeddings.llama_cpp", fake_llama_cpp), \
             patch("os.path.isfile", return_value=True):
            from ovos_gguf_embeddings import GGUFEmbeddings
            emb = GGUFEmbeddings(config={"model": "/fake/path/model.gguf"})

        self.assertIsNotNone(emb.model)
        result = emb.get_embeddings("hi")
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, [0.1, 0.2, 0.3])


class TestGGUFEmbeddingsUnknownModel(unittest.TestCase):
    """(b) unknown model id leaves self.model is None and get_embeddings raises RuntimeError."""

    def test_unknown_model_raises(self):
        fake_llama_cpp, _ = _make_fake_llama_cpp()

        with patch("ovos_gguf_embeddings.llama_cpp", fake_llama_cpp), \
             patch("os.path.isfile", return_value=False):
            from ovos_gguf_embeddings import GGUFEmbeddings
            emb = GGUFEmbeddings(config={"model": "not-a-real-model-id"})

        self.assertIsNone(emb.model)
        with self.assertRaises(RuntimeError):
            emb.get_embeddings("hello")


class TestGGUFEmbeddingsDefaultModel(unittest.TestCase):
    """(c) DEFAULT_MODELS name triggers download then loads."""

    def test_default_model_triggers_download(self):
        fake_llama_cpp, fake_model = _make_fake_llama_cpp()

        with patch("ovos_gguf_embeddings.llama_cpp", fake_llama_cpp), \
             patch("os.path.isfile", return_value=False), \
             patch("ovos_gguf_embeddings.GGUFEmbeddings._download_model") as mock_dl:
            from ovos_gguf_embeddings import GGUFEmbeddings
            emb = GGUFEmbeddings(config={"model": "all-MiniLM-L6-v2"})

        mock_dl.assert_called_once()
        self.assertIsNotNone(emb.model)


class TestGGUFEmbeddingsExtraConfig(unittest.TestCase):
    """(d) extra config keys forwarded to Llama constructor; 'model' key not forwarded."""

    def test_extra_kwargs_forwarded(self):
        fake_llama_cpp, fake_model = _make_fake_llama_cpp()

        with patch("ovos_gguf_embeddings.llama_cpp", fake_llama_cpp), \
             patch("os.path.isfile", return_value=True):
            from ovos_gguf_embeddings import GGUFEmbeddings
            GGUFEmbeddings(config={
                "model": "/fake/model.gguf",
                "n_gpu_layers": 4,
                "n_ctx": 512,
            })

        call_kwargs = fake_llama_cpp.Llama.call_args[1]
        self.assertEqual(call_kwargs["n_gpu_layers"], 4)
        self.assertEqual(call_kwargs["n_ctx"], 512)
        self.assertNotIn("model", call_kwargs)
        self.assertIn("model_path", fake_llama_cpp.Llama.call_args[1])


class TestGGUFEmbeddingsDefaults(unittest.TestCase):
    """(e) DEFAULT_MODELS is non-empty and labse is the default model."""

    def test_default_models_non_empty(self):
        from ovos_gguf_embeddings import GGUFEmbeddings
        self.assertTrue(len(GGUFEmbeddings.DEFAULT_MODELS) > 0)

    def test_labse_is_default(self):
        fake_llama_cpp, _ = _make_fake_llama_cpp()

        with patch("ovos_gguf_embeddings.llama_cpp", fake_llama_cpp), \
             patch("os.path.isfile", return_value=False), \
             patch("ovos_gguf_embeddings.GGUFEmbeddings._download_model") as mock_dl:
            from ovos_gguf_embeddings import GGUFEmbeddings
            GGUFEmbeddings(config={})

        # download was called with the labse URL
        call_args = mock_dl.call_args
        self.assertIn("labse", call_args[0][0])

    def test_labse_in_default_models(self):
        from ovos_gguf_embeddings import GGUFEmbeddings
        self.assertIn("labse", GGUFEmbeddings.DEFAULT_MODELS)


if __name__ == "__main__":
    unittest.main()
