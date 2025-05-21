mod audio;
use pyo3::{prelude::*, wrap_pymodule};
use pyo3::types::PyDict;

/// A Python module implemented in Rust.
#[pymodule(name = "_core")]
fn rust_ext<'py>(python: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(audio::audio_module))?;

    let sys = PyModule::import(python, "sys")?;
    let sys_modules: Bound<'_, PyDict> = sys.getattr("modules")?.downcast_into()?;
    sys_modules.set_item("osc_data.audio", m.getattr("audio")?)?;
    Ok(())
}
