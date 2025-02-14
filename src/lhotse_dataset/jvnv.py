import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import lhotse
from lhotse.supervision import AlignmentItem

from lhotse_dataset.base import BaseCorpus, Gender, Language
from lhotse_dataset.utils import download_file


@dataclass
class Transcription:
    all: str
    non_verbal: str


class JVNV(BaseCorpus):
    @property
    def url(self) -> str:
        return "https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus"

    @property
    def download_url(self) -> str:
        return "https://ss-takashi.sakura.ne.jp/corpus/jvnv/jvnv_ver1.zip"

    @property
    def language(self) -> Language:
        return Language.JA

    def get_cuts(self) -> Generator[lhotse.MonoCut, None, None]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "jvnv.zip"
            download_file(self.download_url, tmp_path)
            jvnv_zip = zipfile.ZipFile(tmp_path)

            transcription_path = "jvnv_v1/transcription.csv"
            assert transcription_path in [
                file_name.filename for file_name in jvnv_zip.filelist
            ]

            with jvnv_zip.open(transcription_path, "r") as transcript_file:
                lines = transcript_file.read().decode("utf-8").splitlines()
            transcriptions: dict[str, Transcription] = {}
            for line in lines:
                utterance, non_verbal, all = line.split("|")
                transcriptions[utterance] = Transcription(all, non_verbal)

            wav_zippaths = [
                file_name.filename
                for file_name in jvnv_zip.filelist
                if file_name.filename.endswith(".wav")
            ]

            for wav_zippath in sorted(wav_zippaths):
                audio_id = Path(wav_zippath).stem
                with jvnv_zip.open(str(wav_zippath), "r") as audio_file:
                    wav_bytes = audio_file.read()
                recording = lhotse.Recording.from_bytes(
                    wav_bytes, f"recording_{audio_id}"
                )
                speaker = audio_id.split("_")[0]
                utterance = "_".join(audio_id.split("_")[1:])
                gender = Gender.MALE if speaker in ["M1", "M2"] else Gender.FEMALE

                nonverbal_duration_info_path = (
                    f"jvnv_v1/nv_label/{speaker}/{audio_id}.txt"
                )
                with jvnv_zip.open(nonverbal_duration_info_path, "r") as duration_file:
                    lines = duration_file.read().decode("utf-8").splitlines()
                start, end, _ = lines[0].split()
                alignment = AlignmentItem(
                    symbol=transcriptions[utterance].non_verbal,
                    start=float(start),
                    duration=float(end) - float(start),
                )

                supervision = lhotse.SupervisionSegment(
                    id=f"transcript_{audio_id}",
                    recording_id=recording.id,
                    start=0,
                    duration=recording.duration,
                    channel=0,
                    text=transcriptions[utterance].all,
                    language=self.language.value,
                    speaker=speaker,
                    gender=gender.value,
                    alignment={"word": [alignment]},
                )
                cut = lhotse.MonoCut(
                    id=f"{audio_id}",
                    start=0,
                    duration=recording.duration,
                    channel=0,
                    supervisions=[supervision],
                    recording=recording,
                )
                yield cut
