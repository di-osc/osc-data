mod audio;

use audio::AudioDoc;
use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn _lowlevel(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AudioDoc>()?;
    Ok(())
}
