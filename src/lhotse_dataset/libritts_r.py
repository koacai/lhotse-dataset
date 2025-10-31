import tarfile
import tempfile
from pathlib import Path
from typing import Generator

import lhotse
import pandas as pd

from lhotse_dataset.base import BaseCorpus, Gender, Language, SpeakerInfo
from lhotse_dataset.utils import download_file


class LibriTTSR(BaseCorpus):
    @property
    def url(self) -> str:
        return "https://www.openslr.org/141"

    @property
    def download_url(self) -> dict[str, str]:
        return {
            "dev_clean": "https://www.openslr.org/resources/141/dev_clean.tar.gz",
            # "dev_other": "https://www.openslr.org/resources/141/dev_other.tar.gz",
            "test_clean": "https://www.openslr.org/resources/141/test_clean.tar.gz",
            # "test_other": "https://www.openslr.org/resources/141/test_other.tar.gz",
            "train_clean_100": "https://www.openslr.org/resources/141/train_clean_100.tar.gz",
            "train_clean_360": "https://www.openslr.org/resources/141/train_clean_360.tar.gz",
            # "train_other_500": "https://www.openslr.org/resources/141/train_other_500.tar.gz",
        }

    @property
    def doc_url(self) -> str:
        return "https://www.openslr.org/resources/141/doc.tar.gz"

    @property
    def language(self) -> Language:
        return Language.EN

    @property
    def shard_size(self) -> int:
        return 5000

    def get_cuts(self) -> Generator[lhotse.MonoCut, None, None]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            tmp_doc_path = tmp_dir_path / "doc.tar.gz"
            download_file(self.doc_url, tmp_doc_path)

            with tarfile.open(tmp_doc_path) as tar:
                tar.extractall(tmp_dir_path, filter="fully_trusted")

            speakers = {}
            speakers_tsv_path = Path(tmp_dir) / "LibriTTS_R/speakers.tsv"
            df_speakers = pd.read_csv(speakers_tsv_path, sep="\t", index_col=0)
            for row in df_speakers.itertuples():
                gender = Gender.MALE if row[1] == "M" else Gender.FEMALE
                speaker_id = str(row[0])
                speakers[speaker_id] = SpeakerInfo(
                    id=speaker_id, name=str(row[3]), gender=gender
                )

            for subset, download_url in self.download_url.items():
                tmp_path = tmp_dir_path / f"{subset}.tar.gz"
                download_file(download_url, tmp_path)

                with tarfile.open(tmp_path) as tar:
                    tar.extractall(tmp_dir_path, filter="fully_trusted")

                wav_files = list(tmp_dir_path.glob("LibriTTS_R/**/*.wav"))
                for wav_file in wav_files:
                    normalized_txt_path = (
                        wav_file.parent / f"{wav_file.stem}.normalized.txt"
                    )

                    try:
                        with open(normalized_txt_path, "r", encoding="utf-8") as f:
                            normalized_txt = f.readline()
                    except FileNotFoundError:
                        print("Warning: Normalized text file not found for", wav_file)
                        normalized_txt = ""

                    original_txt_path = (
                        wav_file.parent / f"{wav_file.stem}.original.txt"
                    )
                    try:
                        with open(original_txt_path, "r", encoding="utf-8") as f:
                            original_txt = f.readline()
                    except FileNotFoundError:
                        print("Warning: Original text file not found for", wav_file)
                        original_txt = ""

                    speaker_id = wav_file.stem.split("_")[0]
                    audio_id = f"libritts_r_{subset}_{wav_file.stem}"

                    recording = lhotse.Recording.from_file(str(wav_file))

                    supervision = lhotse.SupervisionSegment(
                        id=f"segment_{audio_id}",
                        recording_id=recording.id,
                        start=0,
                        duration=recording.duration,
                        channel=0,
                        text=normalized_txt,
                        language=self.language.value,
                        speaker=speaker_id,
                        gender=speakers[speaker_id].gender.value,
                        custom={
                            "subset": subset,
                            "original_text": original_txt,
                            "speaker_name": speakers[speaker_id].name,
                        },
                    )

                    cut = lhotse.MonoCut(
                        id=audio_id,
                        start=0,
                        duration=recording.duration,
                        channel=0,
                        supervisions=[supervision],
                        recording=recording,
                    )
                    yield cut
