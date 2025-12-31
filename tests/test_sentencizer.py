from osc_data.text_stream import TextStreamSentencizer


sentencizer = TextStreamSentencizer()

text = "有个阿姨特别聪明，她同时接两个钟点工单，上午一家下午一家，收入比全职还高！现在客户都抢着要她。记住啊姐妹们，刚开始别挑活，积累经验最重要。"

sents = []
for c in text:
    sents.extend(sentencizer.push(c))
sents.extend(sentencizer.flush())

assert (
    sents[0]
    == "有个阿姨特别聪明，她同时接两个钟点工单，上午一家下午一家，收入比全职还高！"
), sents[0]
assert sents[1] == "现在客户都抢着要她。", sents[1]
assert sents[2] == "记住啊姐妹们，刚开始别挑活，积累经验最重要。", sents[2]
