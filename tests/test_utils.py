"""Native CPU checkpoint loading regression and neighboring routing controls."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from dinov2.utils.utils import load_pretrained_weights


class LoadPretrainedWeightsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        self.model = torch.nn.Linear(3, 2, bias=False)

    def assert_weights_loaded(self):
        torch.testing.assert_close(self.model.weight, self.expected, rtol=0, atol=0)

    def test_local_file_with_url_scheme(self):
        # A Windows drive and a POSIX relative filename can both parse as schemes.
        filename = "checkpoint.pth" if os.name == "nt" else "checkpoint:weights.pth"
        checkpoint = self.directory / filename
        torch.save({"weight": self.expected}, checkpoint)
        source = str(checkpoint) if os.name == "nt" else filename
        previous_directory = os.getcwd()
        try:
            os.chdir(self.directory)
            load_pretrained_weights(self.model, source, checkpoint_key=None)
        finally:
            os.chdir(previous_directory)
        self.assert_weights_loaded()

    def test_relative_local_file(self):
        checkpoint = self.directory / "checkpoint.pth"
        torch.save({"weight": self.expected}, checkpoint)
        load_pretrained_weights(self.model, os.path.relpath(checkpoint), checkpoint_key=None)
        self.assert_weights_loaded()

    def test_checkpoint_key_and_prefixes(self):
        checkpoint = self.directory / "checkpoint.pth"
        torch.save({"teacher": {"module.backbone.weight": self.expected}}, checkpoint)
        load_pretrained_weights(self.model, str(checkpoint), checkpoint_key="teacher")
        self.assert_weights_loaded()

    def test_urls_keep_using_hub_loader(self):
        for scheme in ("https", "http", "file"):
            with self.subTest(scheme=scheme):
                source = scheme + "://example.org/checkpoint.pth"
                with patch("torch.hub.load_state_dict_from_url", return_value={"weight": self.expected}) as loader:
                    load_pretrained_weights(self.model, source, checkpoint_key=None)
                self.assertEqual(loader.call_args.args, (source,))
                self.assertEqual(loader.call_args.kwargs["map_location"], "cpu")
                self.assert_weights_loaded()


if __name__ == "__main__":
    unittest.main()
