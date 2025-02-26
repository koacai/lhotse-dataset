import itertools
import unittest
from pathlib import Path

import numpy as np

from lhotse_dataset.jis import JIS


class TestJIS(unittest.TestCase):
    def setUp(self) -> None:
        root_dir = Path("/absolute/path/to/jis")
        self.jis = JIS(root_dir)

    def test_get_cuts(self):
        gen = self.jis.get_cuts()
        for cut in itertools.islice(gen, 3):
            audio = cut.load_audio()
            self.assertIsInstance(audio, np.ndarray)
            self.assertEqual(cut.sampling_rate, 48_000)
            if isinstance(audio, np.ndarray):
                self.assertAlmostEqual(cut.duration, audio.shape[1] / cut.sampling_rate)


if __name__ == "__main__":
    unittest.main()
