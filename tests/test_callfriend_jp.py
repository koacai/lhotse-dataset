import itertools

import numpy as np

from lhotse_dataset.callfriend_jp import CallFriendJP


def test_callfriend_jp() -> None:
    email = "email"  # NEED TO CHANGE
    pswd = "pswd"  # NEED TO CHANGE
    corpus = CallFriendJP(email, pswd)
    gen = corpus.get_cuts()
    for cut in itertools.islice(gen, 3):
        audio = cut.load_audio()
        assert isinstance(audio, np.ndarray)
        assert cut.sampling_rate == 8_000
        assert cut.duration == audio.shape[1] / cut.sampling_rate
