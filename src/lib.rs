mod audio;
mod text;
mod text_stream;
mod video;
use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule(name = "_core")]
fn rust_ext<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    let _ = audio::register_module(m)?;
    let _ = text_stream::register_module(m)?;
    let _ = text::register_module(m)?;
    let _ = video::register_module(m)?;
    Ok(())
}
