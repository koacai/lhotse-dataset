from lhotse_dataset.base import BaseCorpus, Language


class SeamlessInteraction(BaseCorpus):
    @property
    def url(self) -> str:
        return "https://ai.meta.com/research/seamless-interaction/"

    @property
    def language(self) -> Language:
        return Language.EN

    @property
    def shard_size(self) -> int:
        return 1000
