import tempfile
import zipfile
from pathlib import Path
from typing import Generator

import gdown
import lhotse

from lhotse_dataset.base import BaseCorpus, Gender, Language


class JVS(BaseCorpus):
    @property
    def url(self) -> str:
        return "https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus"

    @property
    def download_url(self) -> str:
        return "https://drive.google.com/uc?id=19oAw8wWn3Y7z6CKChRdAyGOB9yupL_Xt"

    @property
    def language(self) -> Language:
        return Language.JA

    def get_cuts(self) -> Generator[lhotse.MonoCut, None, None]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "jvs.zip"
            gdown.download(self.download_url, str(tmp_path))
            jvs_zip = zipfile.ZipFile(tmp_path)
            transcript_zippaths = [
                file_name.filename
                for file_name in jvs_zip.filelist
                if "transcripts_utf8" in file_name.filename
            ]

            speaker_gender: dict[str, Gender] = {}

            speaker_gender_path = "jvs_ver1/gender_f0range.txt"
            assert speaker_gender_path in [
                file_name.filename for file_name in jvs_zip.filelist
            ], "speaker gender path (%s) is not in zipfile" % speaker_gender_path

            with jvs_zip.open(speaker_gender_path) as speaker_gender_file:
                lines = speaker_gender_file.read().decode("utf-8").splitlines()
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                speaker_id, str_gender, _, _ = line.split()
                speaker_gender[speaker_id] = (
                    Gender.MALE if str_gender == "M" else Gender.FEMALE
                )

            for transcript_zippath in sorted(transcript_zippaths):
                with jvs_zip.open(transcript_zippath) as transcript_file:
                    lines = transcript_file.read().decode("utf-8").splitlines()
                for line in lines:
                    audio_id, text = line.split(":")
                    name, text = line.split(":")
                    transcript_path = Path(transcript_zippath)
                    utter_type = transcript_path.parent.name
                    speaker_id = transcript_path.parent.parent.name
                    audio_path = (
                        transcript_path.parent / "wav24kHz16bit" / f"{name}.wav"
                    )

                    audio_id = f"{speaker_id}_{utter_type}_{name}"
                    try:
                        with jvs_zip.open(str(audio_path), "r") as audio_file:
                            wav_bytes = audio_file.read()
                        recording = lhotse.Recording.from_bytes(
                            wav_bytes, f"recording_{audio_id}"
                        )
                        supervision = lhotse.SupervisionSegment(
                            id=f"transcript_{audio_id}",
                            recording_id=recording.id,
                            start=0,
                            duration=recording.duration,
                            channel=0,
                            text=text,
                            language=self.language.value,
                            speaker=speaker_id,
                            gender=speaker_gender[speaker_id].value,
                            custom={"utter_type": utter_type},
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
                    except KeyError:
                        # NOTE: transcriptファイルにあるのに音源が無いものがある
                        continue
