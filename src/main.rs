mod text_stream;
use text_stream::ChineseSentenceSplitter;

fn main() {
    let mut splitter = ChineseSentenceSplitter::new(10);
    let text = "你好！这是一个测试，看看能不能分句。继续测试...";
    let sentences = splitter.process_text(text, true, None);
    for s in sentences {
        println!("{}", s);
    }
}
