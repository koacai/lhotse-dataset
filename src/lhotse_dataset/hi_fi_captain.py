import tempfile
import zipfile
from pathlib import Path
from typing import Generator

import lhotse

from lhotse_dataset.base import BaseCorpus, Gender, Language
from lhotse_dataset.utils import download_file


class HiFiCAPTAIN(BaseCorpus):
    @property
    def url(self) -> str:
        return "https://ast-astrec.nict.go.jp/release/hi-fi-captain/"

    @property
    def download_url(self) -> dict[str, str]:
        return {
            "en-US_F": "https://ast-astrec.nict.go.jp/release/hi-fi-captain/hfc_en-US_F.zip",
            "en-US_M": "https://ast-astrec.nict.go.jp/release/hi-fi-captain/hfc_en-US_M.zip",
            "ja-JP_F": "https://ast-astrec.nict.go.jp/release/hi-fi-captain/hfc_ja-JP_F.zip",
            "ja-JP_M": "https://ast-astrec.nict.go.jp/release/hi-fi-captain/hfc_ja-JP_M.zip",
        }

    def get_cuts(self) -> Generator[lhotse.MonoCut, None, None]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for lang_gender, download_url in self.download_url.items():
                zip_path = Path(tmp_dir) / f"{lang_gender}.zip"

                if not zip_path.exists():
                    download_file(download_url, zip_path)

                zip_file = zipfile.ZipFile(zip_path)

                transcript_zippaths = [
                    file_name.filename
                    for file_name in zip_file.filelist
                    if file_name.filename.endswith(".txt")
                ]

                for transcript_zippath in sorted(transcript_zippaths):
                    with zip_file.open(transcript_zippath) as transcript_file:
                        lines = transcript_file.read().decode("utf-8").splitlines()

                    for line in lines:
                        name = line.split()[0]
                        text = " ".join(line.split()[1:])
                        dataset_type = Path(transcript_zippath).stem

                        audio_id = f"hfc_{lang_gender}_{dataset_type}_{name}"

                        audio_path = (
                            Path(transcript_zippath).parent.parent
                            / "wav"
                            / f"{dataset_type}"
                            / f"{name}.wav"
                        )

                        with zip_file.open(str(audio_path), "r") as audio_file:
                            wav_bytes = audio_file.read()

                        recording = lhotse.Recording.from_bytes(
                            wav_bytes, f"recording_{audio_id}"
                        )

                        language = (
                            Language.EN
                            if lang_gender in ["en-US_F", "en-US_M"]
                            else Language.JA
                        )
                        gender = (
                            Gender.MALE
                            if lang_gender in ["en-US_M", "ja-JP_M"]
                            else Gender.FEMALE
                        )

                        supervision = lhotse.SupervisionSegment(
                            id=f"transcript_{audio_id}",
                            recording_id=recording.id,
                            start=0,
                            duration=recording.duration,
                            channel=0,
                            text=text,
                            language=language.value,
                            speaker=f"hfc_{lang_gender}",
                            gender=gender.value,
                            custom={"dataset_type": dataset_type},
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
