import itertools

import numpy as np

from lhotse_dataset.daily_talk import DailyTalk


def test_daily_talk() -> None:
    corpus = DailyTalk()
    gen = corpus.get_cuts()
    for cut in itertools.islice(gen, 3):
        audio = cut.load_audio()
        assert isinstance(audio, np.ndarray)
        assert cut.sampling_rate == 44_100
        assert cut.duration == audio.shape[1] / cut.sampling_rate
