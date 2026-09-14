"""Regression coverage for canonical setup-state paths, including Windows aliases."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from yue2_modly.common import EXTENSION_ID, write_json
from yue2_modly.paths import resolve_models_root


class SavedPathIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="YuE2 paths ñ ")
        self.addCleanup(self.tmp.cleanup)
        # Keep the raw tempfile spelling: Windows may supply a RUNNER~1 alias.
        self.base = Path(self.tmp.name)
        self.root = self.base / "extensions" / EXTENSION_ID
        self.root.mkdir(parents=True)
        self.state = self.root / ".setup-state.json"
        # This directory need not exist yet; resolving it must not create it.
        self.models = self.base / "models"

    def save(self, extension_root, models_root=None):
        write_json(self.state, {
            "extension_root": extension_root,
            "models_root": str(self.models if models_root is None else models_root),
        })

    def resolve(self):
        return resolve_models_root(
            root=self.root, state_file=self.state,
            env={"APPDATA": str(self.base / "roaming"),
                 "XDG_CONFIG_HOME": str(self.base / "config")},
        )

    def equivalent_root(self):
        # A real, noncanonical spelling reproducible on every platform, without mocks.
        return self.root.parent / ".." / self.root.parent.name / self.root.name

    def test_saved_state_accepts_equivalent_absolute_spelling(self):
        self.save(str(self.equivalent_root()))
        self.assertEqual(self.resolve(), self.models.resolve())
        self.assertFalse(self.models.exists())

    def test_saved_state_accepts_raw_tempfile_spelling(self):
        self.save(str(self.root))
        self.assertEqual(self.resolve(), self.models.resolve())

    def test_saved_state_rejects_malformed_or_relative_root(self):
        for value in (None, "", 7, [], {}, "relative/extensions", str(self.root) + "\0"):
            with self.subTest(value=value):
                self.save(value)
                with self.assertRaisesRegex(ValueError, "MODELS_DIR_MISSING"):
                    self.resolve()
        self.assertFalse(self.models.exists())

    def test_saved_state_rejects_other_canonical_directory(self):
        other = self.root.parent / "another-extension"
        other.mkdir()
        self.save(str(other / ".." / other.name))
        with self.assertRaisesRegex(ValueError, "MODELS_DIR_MISSING"):
            self.resolve()
        self.assertFalse(self.models.exists())

    def test_alias_does_not_allow_models_inside_extension(self):
        self.save(str(self.equivalent_root()), self.root / "models")
        with self.assertRaisesRegex(ValueError, "models_dir must be separate"):
            self.resolve()
        self.assertFalse((self.root / "models").exists())

    @unittest.skipUnless(os.name == "nt", "Native Windows path semantics")
    def test_windows_case_and_separator_variants(self):
        canonical = str(self.root.resolve())
        for spelling in (canonical.upper(), canonical.replace("\\", "/")):
            with self.subTest(spelling=spelling):
                self.save(spelling)
                self.assertEqual(self.resolve(), self.models.resolve())

    @unittest.skipUnless(os.name == "nt", "Requires Windows GetShortPathNameW")
    def test_windows_short_and_long_names_bind_same_state(self):
        import ctypes
        from ctypes import wintypes

        get_short_path = ctypes.WinDLL("kernel32", use_last_error=True).GetShortPathNameW
        get_short_path.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD)
        get_short_path.restype = wintypes.DWORD
        canonical = self.root.resolve()
        size = get_short_path(str(canonical), None, 0)
        if not size:
            raise ctypes.WinError(ctypes.get_last_error())
        buffer = ctypes.create_unicode_buffer(size)
        written = get_short_path(str(canonical), buffer, size)
        if not written:
            raise ctypes.WinError(ctypes.get_last_error())
        self.assertLess(written, size)
        short_path = Path(buffer.value)
        if short_path == canonical:
            self.skipTest("This volume does not expose a distinct 8.3 alias")
        self.assertTrue(short_path.samefile(canonical))
        for spelling in (str(short_path), str(canonical)):
            with self.subTest(spelling=spelling):
                self.save(spelling)
                self.assertEqual(self.resolve(), self.models.resolve())


if __name__ == "__main__":
    unittest.main()
