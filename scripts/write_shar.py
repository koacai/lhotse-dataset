import argparse
from pathlib import Path

from lhotse.shar import SharWriter
from tqdm import tqdm

from lhotse_dataset import JIS, JSUT, JVNV, JVS, HQYouTube

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--hq_youtube_tar_path", type=str, required=False)
    parser.add_argument("--jis_dir", type=str, required=False)
    args = parser.parse_args()

    if args.corpus == "jvs":
        corpus = JVS()
    elif args.corpus == "hq_youtube":
        corpus = HQYouTube(args.hq_youtube_tar_path)
    elif args.corpus == "jvnv":
        corpus = JVNV()
    elif args.corpus == "jis":
        corpus = JIS(Path(args.jis_dir))
    elif args.corpus == "jsut":
        corpus = JSUT()
    else:
        raise ValueError(f"invalid corpus name: {args.corpus}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with SharWriter(str(output_dir), fields={"recording": "flac"}) as writer:
        for cut in tqdm(corpus.get_cuts()):
            writer.write(cut)
