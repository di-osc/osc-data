use ndarray::prelude::*;
use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use uuid::Uuid;

#[derive(Debug, Clone)]
#[pyclass(subclass)]
pub struct Audio {
    #[pyo3(get)]
    id: String,
    #[pyo3(get, set)]
    path: String,
    desc: Option<String>,
    duration: Option<f32>,
    mono: Option<bool>,
    sample_rate: Option<u16>,
    waveform: Option<Array2<f32>>,
}

#[pymethods]
impl Audio {
    #[new]
    fn new(path: String, desc: Option<String>) -> Self {
        let id = Uuid::new_v4().to_string();
        Self {
            id,
            path,
            desc,
            duration: None,
            mono: None,
            sample_rate: None,
            waveform: None,
        }
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
    fn sample_rate(&self) -> Option<u16> {
        self.sample_rate
    }

    #[setter]
    fn set_sample_rate(&mut self, sample_rate: Option<u16>) {
        self.sample_rate = sample_rate;
    }

    #[getter]
    fn desc(&self) -> Option<&str> {
        self.desc.as_deref()
    }

    #[setter]
    fn set_desc(&mut self, desc: Option<String>) {
        self.desc = desc;
    }

    #[getter]
    fn waveform<'py>(&mut self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f32>>> {
        match &mut self.waveform {
            Some(waveform) => Some(waveform.to_pyarray_bound(py)),
            None => None,
        }
    }

    #[setter]
    fn set_waveform(&mut self, waveform: PyReadonlyArray2<f32>) {
        self.waveform = Some(waveform.as_array().to_owned());
    }
}

#[pyfunction]
pub fn low_frame_rate<'py>(
    py: Python<'py>,
    frames: PyReadonlyArray2<f32>,
    m: usize,
    n: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let frames = frames.as_array();
    let n_frames = frames.shape()[0];
    let n_hidden = frames.shape()[1];
    let n_output_frames = n_frames.div_ceil(n);
    let n_output_hidden = n_hidden * m;
    let mut output_frames = Array2::<f32>::zeros((n_output_frames, n_output_hidden));
    // repeat time = (m - 1) // 2
    let repeat_times = (m - 1) / 2;
    // 第一帧填充: 数组第一个元素 * repeat_times + 1 -> frames[1: repeat_times + 1]
    for i in 0..repeat_times + 1 {
        output_frames
            .slice_mut(s![0, i * n_hidden..(i + 1) * n_hidden])
            .assign(&frames);
    }
    for i in 1..(m - repeat_times + 1) {
        output_frames
            .slice_mut(s![0, i * n_hidden..(i + 1) * n_hidden])
            .assign(&frames.slice(s![0..n_frames - i, ..]));
    }

    // 左填充为数组第一个元素 填充repeat_times次
    output_frames
        .slice_mut(s![.., 0..n_hidden])
        .assign(&frames.slice(s![0, ..]));
    // let left_padding = Array2::from_elem((repeat_times, frames.shape()[1]), frames.slice(s![0, ..]));
    Ok(output_frames.to_pyarray_bound(py))
}
