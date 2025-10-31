import argparse
from pathlib import Path

from lhotse import CutSet
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", type=str, required=True)
    args = parser.parse_args()

    cut_paths = sorted(list(map(str, Path(args.corpus_dir).glob("**/cuts.*.jsonl.gz"))))
    recording_paths = sorted(
        list(map(str, Path(args.corpus_dir).glob("**/recording.*.tar")))
    )

    cuts = CutSet.from_shar({"cuts": cut_paths, "recording": recording_paths})
    print(cuts)

    duration = 0
    for cut in tqdm(cuts.data):
        duration += cut.duration

    print(f"{duration / 3600} hours")
