import itertools
import unittest

import numpy as np

from lhotse_dataset.jvnv import JVNV


class TestJVNV(unittest.TestCase):
    def setUp(self) -> None:
        self.jvnv = JVNV()

    def test_get_audio_text_pair(self):
        gen = self.jvnv.get_cuts()
        for cut in itertools.islice(gen, 3):
            audio = cut.load_audio()
            self.assertIsInstance(audio, np.ndarray)
            self.assertEqual(cut.sampling_rate, 48_000)
            if isinstance(audio, np.ndarray):
                self.assertAlmostEqual(cut.duration, audio.shape[1] / cut.sampling_rate)


if __name__ == "__main__":
    unittest.main()
