from pathlib import Path

from kaldifst import TextNormalizer as KaldiTextNormalizer
from pydantic import BaseModel, model_validator, ConfigDict

from ._core import text as core_text
from ._core import reorder as core_reorder


KALDI_TAGGER = KaldiTextNormalizer(
    str(Path(__file__).parent / "assets" / "text" / "tn" / "tagger.fst")
)
# 使用 Rust 实现的 Reorder 类
REORDER = core_reorder.Reorder()
VERALIZER = KaldiTextNormalizer(
    str(Path(__file__).parent / "assets" / "text" / "tn" / "verbalizer.fst")
)
VERALIZER_REMOVE_ERHUA = KaldiTextNormalizer(
    str(
        Path(__file__).parent / "assets" / "text" / "tn" / "verbalizer_remove_erhua.fst"
    )
)


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
        if self.remove_erhua:
            self.tagger = KALDI_TAGGER
            self.verbalizer = VERALIZER_REMOVE_ERHUA
        else:
            self.tagger = KALDI_TAGGER
            self.verbalizer = VERALIZER
        self.reorder = REORDER
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
