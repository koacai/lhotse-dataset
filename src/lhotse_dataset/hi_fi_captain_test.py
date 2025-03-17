import itertools
import unittest

import numpy as np

from lhotse_dataset.hi_fi_captain import HiFiCAPTAIN


class TestHiFiCAPTAIN(unittest.TestCase):
    def setUp(self) -> None:
        self.hificaptain = HiFiCAPTAIN()

    def test_get_cuts(self):
        gen = self.hificaptain.get_cuts()
        for cut in itertools.islice(gen, 3):
            audio = cut.load_audio()
            self.assertIsInstance(audio, np.ndarray)
            self.assertEqual(cut.sampling_rate, 48_000)
            if isinstance(audio, np.ndarray):
                self.assertAlmostEqual(cut.duration, audio.shape[1] / cut.sampling_rate)


if __name__ == "__main__":
    unittest.main()
