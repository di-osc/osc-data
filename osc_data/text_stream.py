from typing import List
from ._core import text_stream as core_text_stream


class TextStreamSentencizer:
    def __init__(
        self,
        min_sentence_length: int = 10,
        use_level2_threshold: int = 50,
        use_level3_threshold: int = 100,
        l1_ends: List[str] = ["!", "?", "。", "？", "！", "；", ";"],
        l2_ends: List[str] = ["、", ",", "，"],
        l3_ends: List[str] = [":", "："],
    ):
        """
        文本流分句器
        Args:
            min_sentence_length (int, optional): 最小句子长度. Defaults to 10.
            use_level2_threshold (int, optional): 使用l2_ends的阈值. Defaults to 50.
            use_level3_threshold (int, optional): 使用l3_ends的阈值. Defaults to 100.
            l1_ends (List[str], optional): 优先级最高的切分字符. Defaults to ["!", "?", "。", "？", "！", "；", ";"].
            l2_ends (List[str], optional): l2优先级切分字符，只有l1切分后无句子并且达到l2阈值才会切分. Defaults to ["、", ",", "，"].
            l3_ends (List[str], optional): l3优先级切分字符,规则同l2. Defaults to [":", "："].
        """
        super().__init__()
        assert check_all_chars(l1_ends), "l1_ends must be a list of chars"
        assert check_all_chars(l2_ends), "l2_ends must be a list of chars"
        assert check_all_chars(l3_ends), "l3_ends must be a list of chars"
        self._sentencizer = core_text_stream.TextStreamSentencizer(
            min_sentence_length=min_sentence_length,
            use_level2_threshold=use_level2_threshold,
            use_level3_threshold=use_level3_threshold,
            l1_ends=l1_ends,
            l2_ends=l2_ends,
            l3_ends=l3_ends,
        )

    def push(self, text: str) -> List[str]:
        return self._sentencizer.push(text)

    def flush(self) -> List[str]:
        return self._sentencizer.flush()

    @property
    def min_sentence_length(self) -> int:
        return self._sentencizer.min_sentence_length

    @min_sentence_length.setter
    def min_sentence_length(self, value: int):
        self._sentencizer.min_sentence_length = value

    @property
    def use_level2_threshold(self) -> int:
        return self._sentencizer.use_level2_threshold

    @use_level2_threshold.setter
    def use_level2_threshold(self, value: int):
        self._sentencizer.use_level2_threshold = value

    @property
    def use_level3_threshold(self) -> int:
        return self._sentencizer.use_level3_threshold

    @use_level3_threshold.setter
    def use_level3_threshold(self, value: int):
        self._sentencizer.use_level3_threshold = value

    @property
    def l1_ends(self) -> List[str]:
        return self._sentencizer.level1_endings

    @l1_ends.setter
    def l1_ends(self, value: List[str]):
        assert check_all_chars(value), "l1_ends must be a list of chars"
        self._sentencizer.level1_endings = value

    @property
    def l2_ends(self) -> List[str]:
        return self._sentencizer.level2_endings

    @l2_ends.setter
    def l2_ends(self, value: List[str]):
        assert check_all_chars(value), "l2_ends must be a list of chars"
        self._sentencizer.level2_endings = value

    @property
    def l3_ends(self) -> List[str]:
        return self._sentencizer.level3_endings

    @l3_ends.setter
    def l3_ends(self, value: List[str]):
        assert check_all_chars(value), "l3_ends must be a list of chars"
        self._sentencizer.level3_endings = value


def check_all_chars(text: List[str]) -> bool:
    for char in text:
        if len(char) != 1 or not isinstance(char, str):
            return False
    return True
