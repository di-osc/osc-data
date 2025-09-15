use anyhow::{anyhow, Context, Result};
use numpy::{PyArray4, PyUntypedArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde::Deserialize;
use std::io::Read;
use std::process::{Command, Stdio};

#[derive(Debug, Deserialize)]
struct FfprobeStreams {
    streams: Vec<FfStream>,
    #[serde(default)]
    format: Option<FfFormat>,
}

#[derive(Debug, Deserialize)]
struct FfStream {
    #[serde(default)]
    width: Option<usize>,
    #[serde(default)]
    height: Option<usize>,
    #[serde(default)]
    avg_frame_rate: Option<String>,
    #[serde(default)]
    r_frame_rate: Option<String>,
    #[serde(default)]
    duration: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FfFormat {
    #[serde(default)]
    duration: Option<String>,
}

#[derive(Debug, Clone)]
struct VideoMeta {
    width: usize,
    height: usize,
    fps: f64,
    duration: f64,
}

fn parse_rate(rate: &str) -> Option<f64> {
    if rate.trim().is_empty() {
        return None;
    }
    let mut parts = rate.split('/');
    let num: f64 = parts.next()?.parse().ok()?;
    match parts.next() {
        Some(den) => {
            let den: f64 = den.parse().ok()?;
            if den == 0.0 { None } else { Some(num / den) }
        }
        None => Some(num),
    }
}

fn parse_f64_opt(s: &Option<String>) -> Option<f64> {
    s.as_ref()?.parse::<f64>().ok()
}

fn ffprobe_meta(input: &str) -> Result<VideoMeta> {
    let output = Command::new("ffprobe")
        .arg("-v").arg("error")
        .arg("-select_streams").arg("v:0")
        .arg("-show_entries")
        .arg("stream=width,height,avg_frame_rate,r_frame_rate,duration:format=duration")
        .arg("-of").arg("json")
        .arg(input)
        .output()
        .with_context(|| "failed to execute ffprobe")?;
    if !output.status.success() {
        return Err(anyhow!("ffprobe failed: {}", String::from_utf8_lossy(&output.stderr)));
    }
    let parsed: FfprobeStreams = serde_json::from_slice(&output.stdout)
        .with_context(|| "failed to parse ffprobe json")?;
    let stream = parsed.streams.into_iter().next().ok_or_else(|| anyhow!("no video stream found"))?;
    let width = stream.width.ok_or_else(|| anyhow!("missing width"))?;
    let height = stream.height.ok_or_else(|| anyhow!("missing height"))?;
    let fps = stream
        .avg_frame_rate.as_ref().and_then(|r| parse_rate(r))
        .or_else(|| stream.r_frame_rate.as_ref().and_then(|r| parse_rate(r)))
        .ok_or_else(|| anyhow!("missing fps"))?;
    let duration = parse_f64_opt(&stream.duration)
        .or_else(|| parsed.format.and_then(|f| parse_f64_opt(&f.duration)))
        .unwrap_or(0.0);
    Ok(VideoMeta { width, height, fps, duration })
}

fn ffmpeg_decode_rgb(input: &str, width: usize, height: usize) -> Result<Vec<u8>> {
    let mut child = Command::new("ffmpeg")
        .arg("-v").arg("error")
        .arg("-nostdin")
        .arg("-i").arg(input)
        .arg("-map").arg("0:v:0")
        .arg("-f").arg("rawvideo")
        .arg("-pix_fmt").arg("rgb24")
        .arg("-vsync").arg("0")
        .arg("-")
        .stdout(Stdio::piped())
        .spawn()
        .with_context(|| "failed to spawn ffmpeg")?;
    let mut out = Vec::new();
    child.stdout.as_mut().ok_or_else(|| anyhow!("no stdout"))?.read_to_end(&mut out)
        .with_context(|| "failed to read ffmpeg stdout")?;
    let status = child.wait().with_context(|| "failed to wait ffmpeg")?;
    if !status.success() {
        return Err(anyhow!("ffmpeg failed"));
    }
    let frame_size = width.checked_mul(height).ok_or_else(|| anyhow!("overflow"))?
        .checked_mul(3).ok_or_else(|| anyhow!("overflow"))?;
    let remainder = out.len() % frame_size;
    if remainder != 0 { out.truncate(out.len() - remainder); }
    Ok(out)
}

fn load_impl<'py>(py: Python<'py>, input: &str) -> PyResult<(Bound<'py, PyArray4<u8>>, f64, f64, usize, usize, usize)> {
    let meta = ffprobe_meta(input).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let bytes = ffmpeg_decode_rgb(input, meta.width, meta.height)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let frame_size = meta.width * meta.height * 3;
    if frame_size == 0 { return Err(PyRuntimeError::new_err("invalid frame size")); }
    let num_frames = bytes.len() / frame_size;
    let array = ndarray::Array4::from_shape_vec((num_frames, meta.height, meta.width, 3usize), bytes)
        .map_err(|e| PyRuntimeError::new_err(format!("failed to build ndarray: {}", e)))?;
    let py_arr = PyArray4::from_owned_array(py, array);
    Ok((py_arr, meta.fps, meta.duration, meta.width, meta.height, num_frames))
}

#[pyfunction]
pub fn load_from_path<'py>(py: Python<'py>, path: &str) -> PyResult<(Bound<'py, PyArray4<u8>>, f64, f64, usize, usize, usize)> {
    load_impl(py, path)
}

#[pyfunction]
pub fn load_from_url<'py>(py: Python<'py>, url: &str) -> PyResult<(Bound<'py, PyArray4<u8>>, f64, f64, usize, usize, usize)> {
    load_impl(py, url)
}

pub fn register_module(core_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let video_module = PyModule::new(core_module.py(), "video")?;
    video_module.add_function(wrap_pyfunction!(load_from_path, &video_module)?)?;
    video_module.add_function(wrap_pyfunction!(load_from_url, &video_module)?)?;
    core_module.add_submodule(&video_module)?;
    Ok(())
}

