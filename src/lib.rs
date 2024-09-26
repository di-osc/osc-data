mod audio;

use audio::Audio;
use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn _lowlevel(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    let audio_module = PyModule::new_bound(py, "audio")?;
    audio_module.add_class::<Audio>()?;
    m.add_submodule(&audio_module)?;
    Ok(())
}
