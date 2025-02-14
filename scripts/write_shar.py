import argparse
from pathlib import Path

from lhotse.shar import SharWriter
from tqdm import tqdm

from lhotse_dataset import JVS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    if args.corpus == "jvs":
        corpus = JVS()
    else:
        raise ValueError(f"invalid corpus name: {args.corpus}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with SharWriter(str(output_dir), fields={"recording": "flac"}) as writer:
        for cut in tqdm(corpus.get_cuts()):
            writer.write(cut)
