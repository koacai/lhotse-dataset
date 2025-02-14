import itertools
import unittest

import numpy as np

from lhotse_dataset.jvs import JVS


class TestJVS(unittest.TestCase):
    def setUp(self) -> None:
        self.jvs = JVS()

    def test_get_audio_text_pair(self):
        gen = self.jvs.get_cuts()
        for cut in itertools.islice(gen, 3):
            audio = cut.load_audio()
            self.assertIsInstance(audio, np.ndarray)
            self.assertEqual(cut.sampling_rate, 24_000)
            if isinstance(audio, np.ndarray):
                self.assertAlmostEqual(cut.duration, audio.shape[1] / cut.sampling_rate)


if __name__ == "__main__":
    unittest.main()
