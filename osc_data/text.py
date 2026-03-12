from pathlib import Path

from kaldifst import TextNormalizer as KaldiTextNormalizer
from pydantic import BaseModel, model_validator, ConfigDict

from ._core import text as core_text
from ._core import reorder as core_reorder


class TextNormalizer(BaseModel):
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    remove_erhua: bool = False
    remove_emoji: bool = False
    to_half_width: bool = False
    tagger: KaldiTextNormalizer = None
    reorder: core_reorder.Reorder = None
    verbalizer: KaldiTextNormalizer = None

    @model_validator(mode="after")
    def setup_model(self) -> "TextNormalizer":
        if self.tagger is None:
            self.tagger = KaldiTextNormalizer(
                str(Path(__file__).parent / "assets" / "text" / "tn" / "tagger.fst")
            )
        if self.verbalizer is None:
            if self.remove_erhua:
                self.verbalizer = KaldiTextNormalizer(
                    str(
                        Path(__file__).parent
                        / "assets"
                        / "text"
                        / "tn"
                        / "verbalizer_remove_erhua.fst"
                    )
                )
            else:
                self.verbalizer = KaldiTextNormalizer(
                    str(
                        Path(__file__).parent
                        / "assets"
                        / "text"
                        / "tn"
                        / "verbalizer.fst"
                    )
                )
        if self.reorder is None:
            self.reorder = core_reorder.Reorder()
        return self

    def normalize(self, text: str) -> str:
        if self.remove_emoji:
            text = core_text.remove_emojis(text)
        if self.to_half_width:
            text = core_text.to_half_width(text)
        text = self.tagger(text)
        text = self.reorder.reorder(text)
        text = self.verbalizer(text)
        return text


class TextCleaner(BaseModel):
    remove_emoji: bool = False
    to_half_width: bool = False

    def clean(self, text: str) -> str:
        if self.remove_emoji:
            text = core_text.remove_emojis(text)
        if self.to_half_width:
            text = core_text.to_half_width(text)
        return text
