import tempfile
import zipfile
from pathlib import Path
from typing import Generator

from lhotse import MonoCut, Recording
from lhotse.utils import uuid

from lhotse_dataset.base import BaseCorpus
from lhotse_dataset.utils import download_file


class DEMAND(BaseCorpus):
    @property
    def url(self) -> str:
        return "https://zenodo.org/records/1227121"

    @property
    def download_urls(self) -> list[str]:
        return [
            "https://zenodo.org/records/1227121/files/DKITCHEN_48k.zip?download=1",
            "https://zenodo.org/records/1227121/files/DLIVING_48k.zip?download=1",
            "https://zenodo.org/records/1227121/files/DWASHING_48k.zip?download=1",
            "https://zenodo.org/records/1227121/files/NFIELD_48k.zip?download=1",
            "https://zenodo.org/records/1227121/files/NPARK_48k.zip?download=1",
            "https://zenodo.org/records/1227121/files/NRIVER_48k.zip?download=1",
            "https://zenodo.org/records/1227121/files/OHALLWAY_48k.zip?download=1",
            "https://zenodo.org/records/1227121/files/OMEETING_48k.zip?download=1",
            "https://zenodo.org/records/1227121/files/OOFFICE_48k.zip?download=1",
            "https://zenodo.org/records/1227121/files/PCAFETER_48k.zip?download=1",
            "https://zenodo.org/records/1227121/files/PRESTO_48k.zip?download=1",
            "https://zenodo.org/records/1227121/files/PSTATION_48k.zip?download=1",
            "https://zenodo.org/records/1227121/files/SPSQUARE_48k.zip?download=1",
            "https://zenodo.org/records/1227121/files/STRAFFIC_48k.zip?download=1",
            "https://zenodo.org/records/1227121/files/TBUS_48k.zip?download=1",
            "https://zenodo.org/records/1227121/files/TCAR_48k.zip?download=1",
            "https://zenodo.org/records/1227121/files/TMETRO_48k.zip?download=1",
        ]

    def get_cuts(self) -> Generator[MonoCut, None, None]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            for download_url in self.download_urls:
                filename = download_url.split("/")[-1].split("?")[0]

                tmp_path = tmp_dir / filename
                download_file(download_url, tmp_path)

                demand_zip = zipfile.ZipFile(tmp_path)
                wav_zippaths = [
                    file_name.filename
                    for file_name in demand_zip.filelist
                    if file_name.filename.endswith(".wav")
                ]

                for wav_zippath in sorted(wav_zippaths):
                    with demand_zip.open(wav_zippath, "r") as audio_file:
                        wav_bytes = audio_file.read()

                    id = uuid.uuid4().hex
                    recording = Recording.from_bytes(wav_bytes, f"recording_{id}")

                    cut = MonoCut(
                        id=id,
                        start=0,
                        duration=recording.duration,
                        channel=0,
                        recording=recording,
                    )

                    yield cut
