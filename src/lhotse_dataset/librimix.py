import tempfile
from pathlib import Path
from typing import Callable, Generator

import git
import pandas as pd
from lhotse import CutSet, MultiCut
from lhotse.cut import Cut, MixedCut, mix_cuts

from lhotse_dataset.base import BaseCorpus, Language


class LibriMix(BaseCorpus):
    def __init__(self, librispeech_shar_dir: Path, wham_noise_shar_dir: Path) -> None:
        super(LibriMix, self).__init__()
        self.librispeech_shar_dir = librispeech_shar_dir
        self.wham_noise_shar_dir = wham_noise_shar_dir

    @property
    def url(self) -> str:
        return "https://github.com/JorisCos/LibriMix"

    @property
    def language(self) -> Language:
        return Language.EN

    def get_cuts(self) -> Generator[MixedCut, None, None]:
        """NOTE: これはsharに書き出すことができない"""

        ls_cut_paths = sorted(
            list(map(str, self.librispeech_shar_dir.glob("cuts.*.jsonl.gz")))
        )
        ls_recording_paths = sorted(
            list(map(str, self.librispeech_shar_dir.glob("recording.*.tar")))
        )
        ls_cuts = CutSet.from_shar(
            {"cuts": ls_cut_paths, "recording": ls_recording_paths}
        )

        wham_cut_paths = sorted(
            list(map(str, self.wham_noise_shar_dir.glob("cuts.*.jsonl.gz")))
        )
        wham_recording_paths = sorted(
            list(map(str, self.wham_noise_shar_dir.glob("recording.*.tar")))
        )
        wham_cuts = CutSet.from_shar(
            {"cuts": wham_cut_paths, "recording": wham_recording_paths}
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            dir = Path(tmp_dir) / "LibriMix"
            git.Repo.clone_from(self.url, dir)

            libri2mix_csv_paths = dir.glob("**/Libri2Mix/*.csv")

            for csv_path in libri2mix_csv_paths:
                df = pd.read_csv(csv_path)
                for row in df.itertuples():
                    source_1_id = self.path_to_librispeech_id(row.source_1_path)  # type: ignore
                    source_1_cut = ls_cuts.filter(self._filter_ls_id(source_1_id))[0]
                    source_1_cut.supervisions[0].custom["gain"] = row.source_1_gain  # type: ignore

                    source_2_id = self.path_to_librispeech_id(row.source_2_path)  # type: ignore
                    source_2_cut = ls_cuts.filter(self._filter_ls_id(source_2_id))[0]
                    source_2_cut.supervisions[0].custom["gain"] = row.source_2_gain  # type: ignore

                    noise_id = self.path_to_noise_id(row.noise_path)  # type: ignore
                    noise_subset_dir = row.noise_path.split("/")[0]  # type: ignore
                    noise_cut = wham_cuts.filter(
                        self._filter_noise_id_subset(noise_id, noise_subset_dir)
                    )[0]
                    noise_cut.custom["gain"] = row.noise_gain  # type: ignore

                    mixed_cut = mix_cuts([source_1_cut, source_2_cut, noise_cut])

                    mixed_cut.id = row.mixture_ID  # type: ignore

                    yield mixed_cut

    @staticmethod
    def path_to_librispeech_id(path: str) -> str:
        subset = path.split("/")[0]
        audio_id = path.split("/")[-1].split(".")[0]
        return f"librispeech_{subset}_{audio_id}"

    @staticmethod
    def path_to_noise_id(path: str) -> str:
        return Path(path).stem

    @staticmethod
    def _filter_ls_id(id: str) -> Callable[[Cut], bool]:
        return lambda cut: cut.id == id

    @staticmethod
    def _filter_noise_id_subset(id: str, subset_dir: str) -> Callable[[MultiCut], bool]:
        if subset_dir == "tr":
            subset = "train"
        elif subset_dir == "cv":
            subset = "validation"
        elif subset_dir == "tt":
            subset = "test"
        else:
            raise ValueError(f"Unknown subset {subset_dir}")

        return (
            lambda cut: cut.id == id
            and cut.custom is not None
            and cut.custom["subset"] == subset
        )
