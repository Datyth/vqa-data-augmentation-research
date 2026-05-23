import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from phrase1_sam3.config import Config
from phrase1_sam3.pipeline import bbox_iou, mask_iou, process_record
from phrase1_sam3.sam_backend import DryRunSamBackend, Sam3Backend, SamInstance


class PipelineTest(unittest.TestCase):
    def test_bbox_iou(self):
        self.assertEqual(bbox_iou([0, 0, 10, 10], [20, 20, 5, 5]), 0.0)
        self.assertGreater(bbox_iou([0, 0, 10, 10], [5, 5, 10, 10]), 0.0)

    def test_mask_iou_uses_binary_pngs(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            left_path = Path(tmp) / "left.png"
            right_path = Path(tmp) / "right.png"
            Image.new("1", (4, 4), 0).save(left_path)
            Image.new("1", (4, 4), 0).save(right_path)
            left = Image.open(left_path)
            right = Image.open(right_path)
            left.putpixel((1, 1), 1)
            left.putpixel((2, 1), 1)
            right.putpixel((2, 1), 1)
            right.putpixel((2, 2), 1)
            left.save(left_path)
            right.save(right_path)
            left.close()
            right.close()

            self.assertAlmostEqual(mask_iou(str(left_path), str(right_path)), 1 / 3)

    def test_sam3_backend_writes_raw_mask_png(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            backend = Sam3Backend(Path(tmp) / "masks")
            backend._image_area = 16
            output = {
                "masks": torch.tensor([[[[False, False, False, False], [False, True, True, False], [False, True, True, False], [False, False, False, False]]]]),
                "boxes": torch.tensor([[1.0, 1.0, 3.0, 3.0]]),
                "scores": torch.tensor([0.9]),
            }
            instances = backend._instances_from_output(Path(tmp) / "image.jpg", "cup", output)
            self.assertEqual(len(instances), 1)
            self.assertEqual(instances[0].bbox, [1, 1, 2, 2])
            self.assertEqual(instances[0].sam_box, [1.0, 1.0, 3.0, 3.0])
            self.assertTrue(Path(instances[0].mask_path).exists())

    def test_process_record_retains_only_final_mask_pngs(self):
        class Backend:
            def __init__(self, mask_dir):
                self.mask_dir = mask_dir

            def segment(self, image_path, prompt):
                self.mask_dir.mkdir(parents=True, exist_ok=True)
                paths = [self.mask_dir / f"mask_{index}.png" for index in range(3)]
                for path in paths:
                    path.write_bytes(b"mask")
                return [
                    SamInstance("low_duplicate", [2, 2, 8, 8], 0.7, 64, 1000, str(paths[0])),
                    SamInstance("kept_duplicate", [2, 2, 8, 8], 0.9, 64, 1000, str(paths[1])),
                    SamInstance("below_threshold", [20, 20, 4, 4], 0.1, 16, 1000, str(paths[2])),
                ]

        with tempfile.TemporaryDirectory() as tmp:
            config = Config(
                datasetlink=Path(tmp),
                imageslink=Path(tmp),
                output_dir=Path(tmp) / "out",
                top_k=1,
                max_images=1,
                mask_threshold=0.5,
                sam_backend="sam3",
            )
            record = SimpleNamespace(
                image_id="1",
                image_path=Path(tmp) / "missing.jpg",
                captions=["A dog sits nearby."],
                questions=[],
                answers=[],
                labels=[],
            )
            artifact = process_record(record, config, Backend(config.output_dir / "masks"))
            objects = artifact["final"]["objects"]
            instances = artifact["raw_sam"]["prompt_results"][0]["instances"]

            self.assertEqual(len(objects), 1)
            self.assertEqual(objects[0]["backend_mask_id"], "kept_duplicate")
            self.assertTrue(Path(objects[0]["mask_path"]).exists())
            self.assertEqual([instance["mask_retained"] for instance in instances], [False, True, False])
            self.assertNotIn("mask_path", instances[0])
            self.assertNotIn("mask_path", instances[2])

    def test_process_record_with_dryrun(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Config(
                datasetlink=Path(tmp),
                imageslink=Path(tmp),
                output_dir=Path(tmp) / "out",
                top_k=2,
                max_images=1,
                mask_threshold=0.5,
                sam_backend="dryrun",
            )
            record = SimpleNamespace(
                image_id="1",
                image_path=Path(tmp) / "missing.jpg",
                captions=["A dog sits on a couch."],
                questions=[],
                answers=[],
                labels=[],
            )
            artifact = process_record(record, config, DryRunSamBackend())
            self.assertEqual(artifact["final"]["image_id"], "1")
            self.assertGreaterEqual(len(artifact["final"]["sam3_prompt_log"]), 1)
            self.assertGreaterEqual(len(artifact["final"]["objects"]), 1)


if __name__ == "__main__":
    unittest.main()
