import itertools
import unittest

import numpy as np

from lhotse_dataset.jsut import JSUT


class TestJVNV(unittest.TestCase):
    def setUp(self) -> None:
        self.jsut = JSUT()

    def test_get_cuts(self):
        gen = self.jsut.get_cuts()
        for cut in itertools.islice(gen, 3):
            audio = cut.load_audio()
            self.assertIsInstance(audio, np.ndarray)
            self.assertEqual(cut.sampling_rate, 48_000)
            if isinstance(audio, np.ndarray):
                self.assertAlmostEqual(cut.duration, audio.shape[1] / cut.sampling_rate)


if __name__ == "__main__":
    unittest.main()
