import itertools

import numpy as np

from lhotse_dataset.daily_talk import DailyTalk


class TestDailyTalk:
    def test_get_cuts(self):
        daily_talk = DailyTalk()
        gen = daily_talk.get_cuts()
        for cut in itertools.islice(gen, 3):
            audio = cut.load_audio()
            assert isinstance(audio, np.ndarray)
            assert cut.sampling_rate == 44_100
            assert cut.duration == audio.shape[1] / cut.sampling_rate
