import struct
from pathlib import Path
import unittest
try:
    import PIL  # noqa: F401
    PIL_AVAILABLE = True
except Exception:
    import types, sys
    pil_stub = types.ModuleType('PIL')
    sys.modules['PIL'] = pil_stub
    sys.modules['PIL.Image'] = types.ModuleType('Image')
    sys.modules['PIL.ImagePalette'] = types.ModuleType('ImagePalette')
    PIL_AVAILABLE = False

try:
    import numpy  # noqa: F401
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False

if NUMPY_AVAILABLE:
    from ofua.dat_tools import DatArchive

@unittest.skipUnless(PIL_AVAILABLE and NUMPY_AVAILABLE, "Dependencies missing")
class TestDatParsing(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path('tests/tmp_dat')
        self.tmpdir.mkdir(parents=True, exist_ok=True)
        self.dat_path = self.tmpdir/'sample.dat'
        data = b'hello'
        name = b'foo.txt'
        name_len = len(name)
        entry = struct.pack('<I', name_len) + name + struct.pack('<B', 0) + struct.pack('<I', len(data)) + struct.pack('<I', len(data)) + struct.pack('<I', 0)
        tree = entry
        tree_size = len(tree)
        with open(self.dat_path, 'wb') as f:
            f.write(data)
            f.write(tree)
            f.write(struct.pack('<I', tree_size))
            f.write(struct.pack('<I', 0))

    def tearDown(self):
        for p in self.tmpdir.rglob('*'):
            p.unlink()
        self.tmpdir.rmdir()

    def test_load_entries(self):
        archive = DatArchive(self.dat_path)
        self.assertTrue(archive.load_entries())
        self.assertIn('foo.txt', archive.entries)
        entry = archive.entries['foo.txt']
        self.assertEqual(entry.packed_size, 5)
        out_dir = self.tmpdir/'out'
        out_dir.mkdir()
        extracted = archive.extract_file('foo.txt', out_dir)
        self.assertTrue(extracted and extracted.exists())
        with open(extracted, 'rb') as f:
            self.assertEqual(f.read(), b'hello')

if __name__ == '__main__':
    unittest.main()
