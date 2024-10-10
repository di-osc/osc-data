mod audio;

use audio::{compute_decibel, low_frame_rate, Audio};
use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn _lowlevel(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    let audio_module = PyModule::new_bound(py, "audio")?;
    audio_module.add_class::<Audio>()?;
    audio_module.add_function(wrap_pyfunction!(low_frame_rate, &audio_module)?)?;
    audio_module.add_function(wrap_pyfunction!(compute_decibel, &audio_module)?)?;
    m.add_submodule(&audio_module)?;
    Ok(())
}
