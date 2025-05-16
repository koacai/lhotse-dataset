from pathlib import Path

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    corpus = hydra.utils.instantiate(cfg.data.corpus)

    output_dir = Path(cfg.data.shar_dir)
    corpus.write_shar(output_dir)


if __name__ == "__main__":
    main()
