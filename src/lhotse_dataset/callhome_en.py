import tempfile
from pathlib import Path
from typing import Generator

import lhotse
import requests
from bs4 import BeautifulSoup

from lhotse_dataset.base import BaseCorpus, Language
from lhotse_dataset.utils import download_file


class CallHomeEn(BaseCorpus):
    def __init__(self, email: str, pswd: str) -> None:
        super(CallHomeEn, self).__init__()
        self.email = email
        self.pswd = pswd

    @property
    def url(self) -> str:
        return "https://ca.talkbank.org/access/CallHome/eng.html"

    @property
    def login_url(self) -> str:
        return "https://sla2.talkbank.org:443/logInUser"

    @property
    def download_page_url(self) -> str:
        return "https://media.talkbank.org/ca/CallHome/eng/0wav"

    @property
    def language(self) -> Language:
        return Language.EN

    @property
    def shard_size(self) -> int:
        return 10

    def get_cuts(self) -> Generator[lhotse.MultiCut, None, None]:
        session = requests.Session()
        headers = {"Content-Type": "application/json"}
        data = {"email": self.email, "pswd": self.pswd}
        response = session.post(self.login_url, headers=headers, json=data)
        if response.status_code != 200:
            raise ValueError(f"{response.status_code}: failed to login")

        response = session.get(self.download_page_url)
        if response.status_code != 200:
            raise ValueError(f"{response.status_code}: falied to get page")

        soup = BeautifulSoup(response.text, "html.parser")
        links = [
            str(a["href"])  # type: ignore
            for a in soup.find_all("a", href=True)
            if a["href"].endswith(".wav?f=save")  # type: ignore
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            for link in links:
                audio_id = f"callhome_en_{Path(link).stem}"
                tmp_path = Path(tmp_dir) / Path(link).name.split("?")[0]
                download_file(link, tmp_path, session)

                recording = lhotse.Recording.from_file(tmp_path)
                supervision_0 = lhotse.SupervisionSegment(
                    id=f"segment_{audio_id}_0",
                    recording_id=recording.id,
                    start=0,
                    duration=recording.duration,
                    channel=0,
                    language=self.language.value,
                )
                supervision_1 = lhotse.SupervisionSegment(
                    id=f"segment_{audio_id}_1",
                    recording_id=recording.id,
                    start=0,
                    duration=recording.duration,
                    channel=1,
                    language=self.language.value,
                )
                cut = lhotse.MultiCut(
                    id=f"{audio_id}",
                    start=0,
                    duration=recording.duration,
                    channel=[0, 1],
                    supervisions=[supervision_0, supervision_1],
                    recording=recording,
                )
                yield cut
