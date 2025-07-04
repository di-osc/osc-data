from pathlib import Path
import string

from kaldifst import TextNormalizer as KaldiTextNormalizer
from pydantic import BaseModel, model_validator, ConfigDict


EOS = "<EOS>"
TN_ORDERS = {
    "date": ["year", "month", "day"],
    "fraction": ["denominator", "numerator"],
    "measure": ["denominator", "numerator", "value"],
    "money": ["value", "currency"],
    "time": ["noon", "hour", "minute", "second"],
}


class Token:
    def __init__(self, name):
        self.name = name
        self.order = []
        self.members = {}

    def append(self, key, value):
        self.order.append(key)
        self.members[key] = value

    def string(self, orders):
        output = self.name + " {"
        if self.name in orders.keys():
            if (
                "preserve_order" not in self.members.keys()
                or self.members["preserve_order"] != "true"
            ):
                self.order = orders[self.name]

        for key in self.order:
            if key not in self.members.keys():
                continue
            output += ' {}: "{}"'.format(key, self.members[key])
        return output + " }"


class TokenParser:
    def __init__(self):
        self.orders = TN_ORDERS

    def load(self, input):
        assert len(input) > 0
        self.index = 0
        self.text = input
        self.char = input[0]
        self.tokens = []

    def read(self):
        if self.index < len(self.text) - 1:
            self.index += 1
            self.char = self.text[self.index]
            return True
        self.char = EOS
        return False

    def parse_ws(self):
        not_eos = self.char != EOS
        while not_eos and self.char == " ":
            not_eos = self.read()
        return not_eos

    def parse_char(self, exp):
        if self.char == exp:
            self.read()
            return True
        return False

    def parse_chars(self, exp):
        ok = False
        for x in exp:
            ok |= self.parse_char(x)
        return ok

    def parse_key(self):
        assert self.char != EOS
        assert self.char not in string.whitespace

        key = ""
        while self.char in string.ascii_letters + "_":
            key += self.char
            self.read()
        return key

    def parse_value(self):
        assert self.char != EOS
        escape = False

        value = ""
        while self.char != '"':
            value += self.char
            escape = self.char == "\\"
            self.read()
            if escape:
                escape = False
                value += self.char
                self.read()
        return value

    def parse(self, input):
        self.load(input)
        while self.parse_ws():
            name = self.parse_key()
            self.parse_chars(" { ")

            token = Token(name)
            while self.parse_ws():
                if self.char == "}":
                    self.parse_char("}")
                    break
                key = self.parse_key()
                self.parse_chars(': "')
                value = self.parse_value()
                self.parse_char('"')
                token.append(key, value)
            self.tokens.append(token)

    def reorder(self, input):
        self.parse(input)
        output = ""
        for token in self.tokens:
            output += token.string(self.orders) + " "
        return output.strip()


KALDI_TAGGER = KaldiTextNormalizer(
    str(Path(__file__).parent / "assets" / "text" / "tn" / "tagger.fst")
)
REORDER = TokenParser()
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
    tagger: KaldiTextNormalizer = None
    reorder: TokenParser = None
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
        text = self.tagger(text)
        text = self.reorder.reorder(text)
        text = self.verbalizer(text)
        return text
