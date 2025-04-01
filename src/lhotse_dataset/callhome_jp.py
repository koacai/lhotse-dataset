from typing import Generator

import lhotse
import requests

from lhotse_dataset.base import BaseCorpus, Language


class CallHomeJP(BaseCorpus):
    def __init__(self, email: str, pswd: str) -> None:
        super(CallHomeJP, self).__init__()
        self.email = email
        self.pswd = pswd

    @property
    def url(self) -> str:
        return "https://ca.talkbank.org/access/CallHome/jpn.html"

    @property
    def login_url(self) -> str:
        return "https://sla2.talkbank.org:443/logInUser"

    @property
    def download_url(self) -> str:
        return "https://media.talkbank.org/ca/CallHome/jpn/0wav"

    @property
    def language(self) -> Language:
        return Language.JA

    def get_cuts(self) -> Generator[lhotse.MonoCut, None, None]:
        session = requests.Session()
        headers = {"Content-Type": "application/json"}
        data = {"email": self.email, "pswd": self.pswd}
        response = session.post(self.login_url, headers=headers, json=data)
        if response.status_code != 200:
            raise ValueError(f"{response.status_code}: login failed")

        print(response.text)
