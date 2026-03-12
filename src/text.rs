use pyo3::prelude::*;
use regex::Regex;

/// Remove emoji characters from text.
///
/// Matches and removes emojis from common Unicode ranges including:
/// - Emoticons (U+1F600-U+1F64F)
/// - Misc symbols and pictographs (U+1F300-U+1F5FF)
/// - Transport and map symbols (U+1F680-U+1F6FF)
/// - Flags (U+1F1E6-U+1F1FF)
/// - Misc symbols (U+2600-U+26FF)
/// - Dingbats (U+2700-U+27BF)
///
/// Args:
///     input: Text containing emojis
///
/// Returns:
///     Text with emojis removed
#[pyfunction]
fn remove_emojis(input: &str) -> String {
    // 匹配常见 emoji 的 Unicode 范围
    let emoji_regex = Regex::new(
        r"[\u{1F600}-\u{1F64F}\
          \u{1F300}-\u{1F5FF}\
          \u{1F680}-\u{1F6FF}\
          \u{1F1E6}-\u{1F1FF}\
          \u{2600}-\u{26FF}\
          \u{2700}-\u{27BF}]",
    )
    .unwrap();
    emoji_regex.replace_all(input, "").to_string()
}

/// Convert full-width characters to half-width.
///
/// Converts full-width characters (全角) commonly used in Chinese/Japanese
/// text to their half-width (半角) equivalents.
///
/// Args:
///     input: Text containing full-width characters
///
/// Returns:
///     Text with full-width characters converted to half-width
#[pyfunction]
fn to_half_width(input: String) -> String {
    input
        .chars()
        .map(|c| match c {
            '\u{3000}' => ' ', // 全角空格
            '\u{FF01}'..='\u{FF5E}' => {
                // 常见全角字符范围（！～）
                std::char::from_u32((c as u32) - 0xFEE0).unwrap_or(c)
            }
            _ => c,
        })
        .collect()
}

pub fn register_module(core_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let text_module = PyModule::new(core_module.py(), "text")?;
    text_module.add_function(wrap_pyfunction!(remove_emojis, &text_module)?)?;
    text_module.add_function(wrap_pyfunction!(to_half_width, &text_module)?)?;
    core_module.add_submodule(&text_module)?;
    Ok(())
}
