use pyo3::prelude::*;
use std::collections::HashSet;

#[derive(Debug, Default)]
pub struct ChineseSentenceSplitter {
    buffer: String,
    min_sentence_length: usize,
    use_level2_threshold: usize,
    use_level3_threshold: usize,
    level1_endings: HashSet<char>,
    level2_endings: HashSet<String>,
    level3_endings: HashSet<String>,
}

impl ChineseSentenceSplitter {
    pub fn new(min_sentence_length: usize) -> Self {
        let mut level1_endings = HashSet::new();
        for c in ['!', '?', '。', '？', '！', '；', ';'] {
            level1_endings.insert(c);
        }

        let mut level2_endings = HashSet::new();
        for s in ["、", "...", "…", ",", "，"] {
            level2_endings.insert(s.to_string());
        }

        let mut level3_endings = HashSet::new();
        for s in [":", "："] {
            level3_endings.insert(s.to_string());
        }

        ChineseSentenceSplitter {
            buffer: String::new(),
            min_sentence_length: min_sentence_length,
            use_level2_threshold: 30,
            use_level3_threshold: 100,
            level1_endings,
            level2_endings,
            level3_endings,
        }
    }

    pub fn with_min_sentence_length(mut self, length: usize) -> Self {
        self.min_sentence_length = length;
        self
    }

    pub fn with_level2_threshold(mut self, threshold: usize) -> Self {
        self.use_level2_threshold = threshold;
        self
    }

    pub fn with_level3_threshold(mut self, threshold: usize) -> Self {
        self.use_level3_threshold = threshold;
        self
    }

    pub fn process_text(
        &mut self,
        text: &str,
        is_last: bool,
        special_text: Option<&str>,
    ) -> Vec<String> {
        self.buffer.push_str(text);

        if let Some(special) = special_text {
            if self.buffer.ends_with(special) {
                return vec![self.buffer.clone()];
            }
        }

        let (sentences, indices) = self.split_sentences();

        if !is_last {
            if !indices.is_empty() {
                let remaining_start = indices.last().unwrap() + 1;
                self.buffer = self.buffer[remaining_start..].to_string();
            }
            sentences
        } else {
            if sentences.is_empty() {
                let result = self.buffer.clone();
                self.buffer.clear();
                return vec![result];
            }

            if *indices.last().unwrap() == self.buffer.len() - 1 {
                self.buffer.clear();
                sentences
            } else {
                let remaining_start = indices.last().unwrap() + 1;
                let remaining = self.buffer[remaining_start..].to_string();
                self.buffer.clear();

                let mut result = sentences;
                result.push(remaining);
                result
            }
        }
    }

    fn split_sentences(&self) -> (Vec<String>, Vec<usize>) {
        let indices = self.get_sentence_end_indices();
        let mut sentences = Vec::new();
        let mut sent_indices = Vec::new();
        let mut start = 0;

        for i in indices {
            let sent = &self.buffer[start..=i];
            if !sent.is_empty() && sent.len() >= self.min_sentence_length {
                sentences.push(sent.to_string());
                sent_indices.push(i);
                start = i + 1;
            }
        }

        (sentences, sent_indices)
    }

    fn is_sentence_end_level1(&self, c: char) -> bool {
        self.level1_endings.contains(&c)
    }

    fn is_sentence_end_level2(&self, text: &str) -> bool {
        self.level2_endings
            .iter()
            .any(|ending| text.ends_with(ending))
    }

    fn is_sentence_end_level3(&self, text: &str) -> bool {
        self.level3_endings
            .iter()
            .any(|ending| text.ends_with(ending))
    }

    fn get_sentence_end_indices(&self) -> Vec<usize> {
        let sents_l1: Vec<usize> = self
            .buffer
            .char_indices()
            .filter_map(|(i, c)| {
                if self.is_sentence_end_level1(c) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        if sents_l1.is_empty() && self.buffer.len() > self.use_level2_threshold {
            let sents_l2: Vec<usize> = self
                .buffer
                .char_indices()
                .filter_map(|(i, _)| {
                    let substr = &self.buffer[..=i];
                    if self.is_sentence_end_level2(substr) {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect();

            if sents_l2.is_empty() && self.buffer.len() > self.use_level3_threshold {
                self.buffer
                    .char_indices()
                    .filter_map(|(i, _)| {
                        let substr = &self.buffer[..=i];
                        if self.is_sentence_end_level3(substr) {
                            Some(i)
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
    core_module.add_submodule(&audio_module)?;
    Ok(())
}
