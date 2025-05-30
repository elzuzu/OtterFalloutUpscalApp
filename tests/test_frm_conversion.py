import struct
from pathlib import Path
import unittest

try:
    import PIL
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    import numpy
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False

if PIL_AVAILABLE and NUMPY_AVAILABLE:
    from ofua.dat_tools import FRMConverter

@unittest.skipUnless(PIL_AVAILABLE and NUMPY_AVAILABLE, "Dependencies missing")
class TestFRMConversion(unittest.TestCase):
    def setUp(self):
        self.tmp = Path('tests/tmp_frm')
        self.tmp.mkdir(parents=True, exist_ok=True)
        # create palette
        self.pal_path = self.tmp/'color.pal'
        with open(self.pal_path, 'wb') as f:
            f.write(bytes(range(256))*3)
        # create simple FRM
        self.frm_path = self.tmp/'test.frm'
        header = bytearray()
        header += struct.pack('<I', 4)  # version
        header += struct.pack('<H', 10)  # fps
        header += struct.pack('<H', 0)   # action frame
        header += struct.pack('<H', 1)   # frames per dir
        header += b'\x00'*12  # shift x
        header += b'\x00'*12  # shift y
        # offsets
        header += struct.pack('<I', 62) + b'\x00'*20
        frame_header = struct.pack('<HHIhh', 2,2,4,0,0)
        pixel = bytes([0,1,2,3])
        with open(self.frm_path, 'wb') as f:
            f.write(header)
            f.write(frame_header)
            f.write(pixel)
        self.conv = FRMConverter()
        self.conv.load_palette(self.pal_path)

    def tearDown(self):
        for p in self.tmp.rglob('*'):
            p.unlink()
        self.tmp.rmdir()

    def test_conversion(self):
        out_png = self.tmp/'png'
        out_png.mkdir()
        pngs = self.conv.frm_to_png(self.frm_path, out_png)
        self.assertTrue(pngs)
        back_dir = self.tmp/'back'
        back_dir.mkdir()
        grouped = {self.frm_path.stem: pngs}
        new_frm = self.conv.png_to_frm(grouped, back_dir, self.frm_path)
        self.assertTrue(new_frm and new_frm.exists())

if __name__ == '__main__':
    unittest.main()
