use pyo3::prelude::*;
use std::collections::HashSet;

const LEVEL1_ENDINGS: [char; 7] = ['!', '?', '。', '？', '！', '；', ';'];
const LEVEL2_ENDINGS: [char; 3] = ['、', ',', '，'];
const LEVEL3_ENDINGS: [char; 2] = [':', '：'];

/// A simple Chinese sentence splitter for text streams.
///
/// This struct is used to split Chinese text into sentences.
/// It keeps a buffer of text and splits it into sentences when it encounters a sentence ending character.
#[derive(Debug, Default)]
#[pyclass]
pub struct TextStreamSentencizer {
    buffer: String,
    #[pyo3(get, set)]
    min_sentence_length: usize,
    #[pyo3(get, set)]
    use_level2_threshold: usize,
    #[pyo3(get, set)]
    use_level3_threshold: usize,
    #[pyo3(get, set)]
    level1_endings: HashSet<char>,
    #[pyo3(get, set)]
    level2_endings: HashSet<char>,
    #[pyo3(get, set)]
    level3_endings: HashSet<char>,
}

#[pymethods]
impl TextStreamSentencizer {
    #[new]
    #[pyo3(signature = (l1_ends=None, l2_ends=None, l3_ends=None, min_sentence_length=10, use_level2_threshold=50, use_level3_threshold=100))]
    pub fn new(
        l1_ends: Option<Vec<char>>,
        l2_ends: Option<Vec<char>>,
        l3_ends: Option<Vec<char>>,
        min_sentence_length: usize,
        use_level2_threshold: usize,
        use_level3_threshold: usize,
    ) -> Self {
        let level1_endings = l1_ends
            .unwrap_or(LEVEL1_ENDINGS.to_vec())
            .into_iter()
            .collect();
        let level2_endings = l2_ends
            .unwrap_or(LEVEL2_ENDINGS.to_vec())
            .into_iter()
            .collect();
        let level3_endings = l3_ends
            .unwrap_or(LEVEL3_ENDINGS.to_vec())
            .into_iter()
            .collect();
        Self {
            buffer: String::new(),
            min_sentence_length,
            use_level2_threshold,
            use_level3_threshold,
            level1_endings,
            level2_endings,
            level3_endings,
        }
    }

    pub fn push(&mut self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }
        self.buffer.push_str(text);

        let (sentences, indices) = self.split_sentences();

        if !indices.is_empty() {
            let remaining_start = indices.last().unwrap() + 1;
            self.buffer = self.buffer[remaining_start..].to_string();
        }
        sentences
    }

    pub fn flush(&mut self) -> Vec<String> {
        let (sentences, indices) = self.split_sentences();
        if sentences.is_empty() {
            let mut result = vec![];
            if self.buffer.chars().count() > 0 {
                result.push(self.buffer.clone());
            }
            self.buffer.clear();
            return result;
        }

        if *indices.last().unwrap() == self.buffer.len() - 1 {
            self.buffer.clear();
            sentences
        } else {
            let remaining_start = indices.last().unwrap() + 1;
            let remaining = self.buffer[remaining_start..].to_string();
            self.buffer.clear();

            let mut result = sentences;
            if remaining.chars().count() > 0 {
                result.push(remaining);
            }
            result
        }
    }

    fn split_sentences(&self) -> (Vec<String>, Vec<usize>) {
        let end_indices = self.get_sentence_end_indices();
        let mut sentences = Vec::new();
        let mut sent_indices = Vec::new();
        let mut start = 0;

        for end in end_indices {
            let sent = &self.buffer[start..=end];
            let sent_length = sent.chars().count();
            if !sent.is_empty() && sent_length >= self.min_sentence_length {
                sentences.push(sent.to_string());
                sent_indices.push(end);
                start = end + 1;
            }
        }

        (sentences, sent_indices)
    }

    fn get_sentence_end_indices(&self) -> Vec<usize> {
        let sents_l1: Vec<usize> = self
            .buffer
            .char_indices()
            .filter_map(|(i, c)| {
                if self.level1_endings.contains(&c) {
                    Some(i + c.len_utf8() - 1)
                } else {
                    None
                }
            })
            .collect();
        let buffer_char_length = self.buffer.chars().count();
        if sents_l1.is_empty() && buffer_char_length > self.use_level2_threshold {
            let sents_l2: Vec<usize> = self
                .buffer
                .char_indices()
                .filter_map(|(i, c)| {
                    if self.level2_endings.contains(&c) {
                        Some(i + c.len_utf8() - 1)
                    } else {
                        None
                    }
                })
                .collect();

            if sents_l2.is_empty() && buffer_char_length > self.use_level3_threshold {
                self.buffer
                    .char_indices()
                    .filter_map(|(i, c)| {
                        if self.level3_endings.contains(&c) {
                            Some(i + c.len_utf8() - 1)
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                sents_l2
            }
        } else {
            sents_l1
        }
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

pub fn register_module(core_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let audio_module = PyModule::new(core_module.py(), "text_stream")?;
    audio_module.add_class::<TextStreamSentencizer>()?;
    core_module.add_submodule(&audio_module)?;
    Ok(())
}
