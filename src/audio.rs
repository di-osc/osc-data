use pyo3::prelude::*;
use uuid::Uuid;

#[pyclass(subclass)]
pub struct AudioDoc {
    id: String,
    url: String,
    duration: Option<f32>,
    mono: Option<bool>,
    sample_rate: Option<i8>,
}

#[pymethods]
impl AudioDoc {
    #[new]
    fn new(
        url: String,
        duration: Option<f32>,
        mono: Option<bool>,
        id: Option<String>,
        sample_rate: Option<i8>,
    ) -> Self {
        let id = match id {
            Some(id) => id,
            None => Uuid::new_v4().to_string(),
        };
        AudioDoc {
            id,
            url,
            duration: duration,
            mono: mono,
            sample_rate: sample_rate,
        }
    }

    #[getter]
    fn id(&self) -> &str {
        &self.id
    }

    #[setter]
    fn set_id(&mut self, id: String) {
        self.id = id;
    }

    #[getter]
    fn url(&self) -> &str {
        &self.url
    }

    #[setter]
    fn set_url(&mut self, url: String) {
        self.url = url;
    }

    #[getter]
    fn duration(&self) -> Option<f32> {
        self.duration
    }

    #[setter]
    fn set_duration(&mut self, duration: Option<f32>) {
        self.duration = duration;
    }

    #[getter]
    fn mono(&self) -> Option<bool> {
        self.mono
    }

    #[setter]
    fn set_mono(&mut self, mono: Option<bool>) {
        self.mono = mono;
    }

    #[getter]
    fn sample_rate(&self) -> Option<i8> {
        self.sample_rate
    }

    #[setter]
    fn set_sample_rate(&mut self, sample_rate: Option<i8>) {
        self.sample_rate = sample_rate;
    }
}
