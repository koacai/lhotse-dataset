import argparse
from pathlib import Path

import pandas as pd
from lhotse import CutSet
from tqdm import tqdm


def main(shar_dir: Path) -> None:
    cut_paths = sorted(list(map(str, shar_dir.glob("cuts.*.jsonl.gz"))))
    recording_paths = sorted(list(map(str, shar_dir.glob("recording.*.tar"))))

    cuts = CutSet.from_shar({"cuts": cut_paths, "recording": recording_paths})
    cuts = cuts.filter(lambda c: c.duration >= 3.0)  # type: ignore
    cuts = cuts.filter(lambda c: c.duration < 20.0)  # type: ignore
    cuts = cuts.filter(lambda c: len(c.supervisions) > 0)  # type: ignore

    subsets_samples = {
        "test_clean": 10000,
        "dev_clean": 10000,
        "train_clean_100": 100000,
        "train_clean_360": 300000,
    }

    metadata_path = Path("metadata")
    metadata_path.mkdir(parents=True, exist_ok=True)

    for subset, samples in subsets_samples.items():
        cuts_subset = cuts.filter(
            lambda s: s.supervisions[0].custom["subset"] == subset  # type: ignore
        )
        cuts_subset = cuts_subset.sort_by_duration()

        metadata = []
        for cut_1 in tqdm(cuts_subset.data):
            for cut_2 in cuts_subset.data:
                if cut_1.duration < cut_2.duration:
                    continue
                if cut_1.supervisions[0].speaker == cut_2.supervisions[0].speaker:
                    continue

                metadata.append(dict(id_1=cut_1.id, id_2=cut_2.id))

        df = pd.DataFrame(metadata)
        df = df.sample(n=samples, random_state=42)
        df.to_csv(metadata_path / f"{subset}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shar_dir", type=str, required=True)
    args = parser.parse_args()
    shar_dir = Path(args.shar_dir)
    main(shar_dir)
