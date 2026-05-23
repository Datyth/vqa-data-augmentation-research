import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from phrase1_sam3.captioning import CaptionBackend, generate_captions_for_records
from phrase1_sam3.config import Config
from phrase1_sam3.data import ImageRecord


class DummyCaptionBackend(CaptionBackend):
    name = "dummy"

    def caption(self, image_path):
        return f"caption for {image_path.name}"


class CaptioningTest(unittest.TestCase):
    def test_generate_captions_json_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "captions.json"
            config = Config(
                datasetlink=Path(tmp),
                imageslink=Path(tmp),
                caption_output_file=str(output),
            )
            records = [
                ImageRecord(
                    image_id="1",
                    image_path=Path(tmp) / "image.jpg",
                    split="val",
                )
            ]
            data = generate_captions_for_records(records, [DummyCaptionBackend()], config)
            self.assertTrue(output.exists())
            self.assertEqual(data["captions"]["1"], ["caption for image.jpg"])
            self.assertEqual(data["metadata"]["caption_count"], 1)


if __name__ == "__main__":
    unittest.main()
