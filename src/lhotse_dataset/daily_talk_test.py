import itertools
import unittest

import numpy as np

from lhotse_dataset.daily_talk import DailyTalk


class TestDailyTalk(unittest.TestCase):
    def setUp(self) -> None:
        self.daily_talk = DailyTalk()

    def test_get_cuts(self):
        gen = self.daily_talk.get_cuts()
        for cut in itertools.islice(gen, 3):
            audio = cut.load_audio()
            self.assertIsInstance(audio, np.ndarray)
            self.assertEqual(cut.sampling_rate, 44_100)
            if isinstance(audio, np.ndarray):
                self.assertAlmostEqual(cut.duration, audio.shape[1] / cut.sampling_rate)


if __name__ == "__main__":
    unittest.main()
