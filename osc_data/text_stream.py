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
        remove_emoji: bool = False,
    ):
        """
        文本流分句器 - 支持分层优先级的实时句子切分

        使用三层优先级策略切分句子：
        1. L1 (最高优先级)：句号、问号等主要结束符
        2. L2 (中等优先级)：逗号、顿号等次要分隔符（当L1无结果且文本长度超过阈值时启用）
        3. L3 (最低优先级)：冒号等弱分隔符（当L1、L2均无结果且文本长度超过阈值时启用）

        Args:
            min_sentence_length: 最小句子长度（字符数），短于此长度的片段不会被输出
            use_level2_threshold: 启用L2切分的缓冲区长度阈值（字符数）
            use_level3_threshold: 启用L3切分的缓冲区长度阈值（字符数）
            l1_ends: L1级切分字符列表（主要句子结束符）
            l2_ends: L2级切分字符列表（次要分隔符）
            l3_ends: L3级切分字符列表（弱分隔符）
            remove_emoji: 是否在切分前移除emoji字符

        Example:
            >>> from osc_data.text_stream import TextStreamSentencizer
            >>>
            >>> # 创建分句器
            >>> sentencizer = TextStreamSentencizer(min_sentence_length=5)
            >>>
            >>> # 流式输入文本
            >>> text = "你好！这是第一句话。这是第二句话。"
            >>>
            >>> # 逐字符推送，获取完成的句子
            >>> sentences = []
            >>> for char in text:
            ...     completed = sentencizer.push(char)
            ...     sentences.extend(completed)
            >>>
            >>> # 刷新缓冲区，获取剩余内容
            >>> sentences.extend(sentencizer.flush())
            >>>
            >>> print(sentences)
            ['你好！', '这是第一句话。', '这是第二句话。']
            >>>
            >>> # 复杂示例：带emoji的长文本
            >>> sentencizer2 = TextStreamSentencizer(min_sentence_length=10, remove_emoji=True)
            >>> long_text = "有个阿姨特别聪明，她同时接两个钟点工单😊，上午一家下午一家，收入比全职还高！现在客户都抢着要她。"
            >>>
            >>> sentences2 = []
            >>> for char in long_text:
            ...     completed = sentencizer2.push(char)
            ...     sentences2.extend(completed)
            >>> sentences2.extend(sentencizer2.flush())
            >>>
            >>> print(sentences2)
            ['有个阿姨特别聪明，她同时接两个钟点工单，上午一家下午一家，收入比全职还高！', '现在客户都抢着要她。']
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
            remove_emoji=remove_emoji,
        )

    def push(self, text: str) -> List[str]:
        """Push text into the stream and get completed sentences.

        This method processes incoming text character by character (or in chunks)
        and returns any completed sentences detected.

        Args:
            text (str): Text to push into the stream.

        Returns:
            List[str]: List of completed sentences detected from the stream.

        Example:
            >>> sentencizer = TextStreamSentencizer()
            >>> sentences = sentencizer.push("Hello world. This is a test.")
            >>> print(sentences)
            ['Hello world.']
        """
        return self._sentencizer.push(text)

    def flush(self) -> List[str]:
        """Flush the buffer and return remaining content as sentences.

        This method should be called when the stream ends to get any
        remaining text that hasn't been returned as a complete sentence.

        Returns:
            List[str]: List of remaining sentences from the buffer.

        Example:
            >>> sentencizer = TextStreamSentencizer()
            >>> sentencizer.push("Hello world")
            >>> remaining = sentencizer.flush()
            >>> print(remaining)
            ['Hello world']
        """
        return self._sentencizer.flush()


def check_all_chars(text: List[str]) -> bool:
    """Check if all elements in the list are single characters.

    Args:
        text (List[str]): List of strings to check.

    Returns:
        bool: True if all elements are single-character strings, False otherwise.
    """
    for char in text:
        if len(char) != 1 or not isinstance(char, str):
            return False
    return True
