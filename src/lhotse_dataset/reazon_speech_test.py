import itertools
import unittest

import numpy as np

from lhotse_dataset.reazon_speech import ReazonSpeech


class TestReazonSpeech(unittest.TestCase):
    def setUp(self) -> None:
        self.reazon_speech = ReazonSpeech("tiny")

    def test_get_cuts(self) -> None:
        gen = self.reazon_speech.get_cuts()
        for cut in itertools.islice(gen, 3):
            audio = cut.load_audio()
            self.assertIsInstance(audio, np.ndarray)
            self.assertEqual(cut.sampling_rate, 16_000)
            if isinstance(audio, np.ndarray):
                self.assertAlmostEqual(cut.duration, audio.shape[1] / cut.sampling_rate)


if __name__ == "__main__":
    unittest.main()
