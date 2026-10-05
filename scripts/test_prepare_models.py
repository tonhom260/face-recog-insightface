"""Integrity and atomic replacement checks; these use no external network."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest

from prepare_models import atomic_install, verify


class ModelIntegrityTests(unittest.TestCase):
    def test_corrupt_download_preserves_existing_asset_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "source", root / "model.onnx"
            source.write_bytes(b"bad")
            destination.write_bytes(b"existing good asset")
            model = {"name": "test", "size": 3, "sha256": hashlib.sha256(b"new").hexdigest()}
            with self.assertRaises(ValueError):
                atomic_install(source, destination, model)
            self.assertEqual(destination.read_bytes(), b"existing good asset")
            self.assertEqual(list(root.glob(".download-*")), [])

    def test_verified_replacement_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "source", root / "nested/model.task"
            source.write_bytes(b"new")
            model = {"name": "test", "size": 3, "sha256": hashlib.sha256(b"new").hexdigest()}
            atomic_install(source, destination, model)
            self.assertTrue(verify(destination, model))
            destination.write_bytes(b"bad")
            self.assertFalse(verify(destination, model))

    def test_restrictive_umask_still_creates_container_readable_public_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "source", root / "models/model.task"
            source.write_bytes(b"new")
            model = {"name": "test", "size": 3, "sha256": hashlib.sha256(b"new").hexdigest()}
            previous = os.umask(0o077)
            try:
                atomic_install(source, destination, model)
            finally:
                os.umask(previous)
            self.assertEqual(destination.parent.stat().st_mode & 0o777, 0o755)
            self.assertEqual(destination.stat().st_mode & 0o777, 0o644)


if __name__ == "__main__":
    unittest.main()
