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
    from ofua.quality_metrics import QualityMetrics

@unittest.skipUnless(PIL_AVAILABLE and NUMPY_AVAILABLE, "Dependencies missing")
class TestQualityMetrics(unittest.TestCase):
    def setUp(self):
        self.tmp = Path('tests/tmp_qm')
        self.tmp.mkdir(parents=True, exist_ok=True)
        img1 = Image.new('RGB', (4,4), color='red')
        img2 = Image.new('RGB', (4,4), color='red')
        self.orig = self.tmp/'orig.png'
        self.up = self.tmp/'up.png'
        img1.save(self.orig)
        img2.save(self.up)

    def tearDown(self):
        for p in self.tmp.rglob('*'):
            p.unlink()
        self.tmp.rmdir()

    def test_metrics(self):
        score = QualityMetrics.calculate_composite_score(self.orig, self.up, 'character')
        self.assertGreaterEqual(score, 0.9)

if __name__ == '__main__':
    unittest.main()
