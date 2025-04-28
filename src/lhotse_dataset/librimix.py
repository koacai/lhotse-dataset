import tempfile
from pathlib import Path
from typing import Callable, Generator

import git
import pandas as pd
from lhotse import CutSet
from lhotse.cut import Cut, MixedCut, mix_cuts

from lhotse_dataset.base import BaseCorpus, Language


class LibriMix(BaseCorpus):
    def __init__(self, librispeech_shar_dir: Path) -> None:
        self.librispeech_shar_dir = librispeech_shar_dir

    @property
    def url(self) -> str:
        return "https://github.com/JorisCos/LibriMix"

    @property
    def language(self) -> Language:
        return Language.EN

    def get_cuts(self) -> Generator[MixedCut, None, None]:
        ls_cut_paths = sorted(
            list(map(str, self.librispeech_shar_dir.glob("cuts.*.jsonl.gz")))
        )
        ls_recording_paths = sorted(
            list(map(str, self.librispeech_shar_dir.glob("recording.*.tar")))
        )

        ls_cuts = CutSet.from_shar(
            {"cuts": ls_cut_paths, "recording": ls_recording_paths}
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            dir = Path(tmp_dir) / "LibriMix"
            git.Repo.clone_from(self.url, dir)

            libri2mix_csv_paths = dir.glob("**/Libri2Mix/*.csv")

            for csv_path in libri2mix_csv_paths:
                df = pd.read_csv(csv_path)
                for row in df.itertuples():
                    source_1_id = self.path_to_librispeech_id(row.source_1_path)  # type: ignore
                    source_1_cut = ls_cuts.filter(self._filter_id(source_1_id))[0]
                    source_1_cut.supervisions[0].custom["gain"] = row.source_1_gain  # type: ignore

                    source_2_id = self.path_to_librispeech_id(row.source_2_path)  # type: ignore
                    source_2_cut = ls_cuts.filter(self._filter_id(source_2_id))[0]
                    source_2_cut.supervisions[0].custom["gain"] = row.source_2_gain  # type: ignore

                    mixed_cut = mix_cuts([source_1_cut, source_2_cut])

                    mixed_cut.id = row.mixture_ID  # type: ignore

                    yield mixed_cut

    @staticmethod
    def path_to_librispeech_id(path: str) -> str:
        subset = path.split("/")[0]
        audio_id = path.split("/")[-1].split(".")[0]
        return f"librispeech_{subset}_{audio_id}"

    @staticmethod
    def _filter_id(id: str) -> Callable[[Cut], bool]:
        return lambda cut: cut.id == id
