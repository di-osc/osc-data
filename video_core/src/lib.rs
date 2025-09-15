use anyhow::{anyhow, Context, Result};
use ndarray::Array4;
use numpy::IntoPyArray;
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
    #[serde(default)]
    nb_frames: Option<String>,
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
    let den_str = parts.next();
    match den_str {
        Some(den) => {
            let den: f64 = den.parse().ok()?;
            if den == 0.0 { None } else { Some(num / den) }
        }
        None => Some(num),
    }
}

fn parse_f64_str(s: &Option<String>) -> Option<f64> {
    s.as_ref()?.parse::<f64>().ok()
}

fn ffprobe_meta(input: &str) -> Result<VideoMeta> {
    let output = Command::new("ffprobe")
        .arg("-v")
        .arg("error")
        .arg("-select_streams")
        .arg("v:0")
        .arg("-show_entries")
        .arg("stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration")
        .arg("-of")
        .arg("json")
        .arg(input)
        .output()
        .with_context(|| "failed to execute ffprobe")?;
    if !output.status.success() {
        return Err(anyhow!("ffprobe failed: {}", String::from_utf8_lossy(&output.stderr)));
    }
    let parsed: FfprobeStreams = serde_json::from_slice(&output.stdout)
        .with_context(|| "failed to parse ffprobe json")?;
    let stream = parsed
        .streams
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("no video stream found"))?;
    let width = stream.width.ok_or_else(|| anyhow!("missing width from ffprobe"))?;
    let height = stream.height.ok_or_else(|| anyhow!("missing height from ffprobe"))?;
    let fps = stream
        .avg_frame_rate
        .as_ref()
        .and_then(|r| parse_rate(r))
        .or_else(|| stream.r_frame_rate.as_ref().and_then(|r| parse_rate(r)))
        .ok_or_else(|| anyhow!("missing fps from ffprobe"))?;
    let duration = parse_f64_str(&stream.duration)
        .or_else(|| parsed.format.and_then(|f| parse_f64_str(&f.duration)))
        .unwrap_or(0.0);
    Ok(VideoMeta {
        width,
        height,
        fps,
        duration,
    })
}

fn ffmpeg_decode_rgb(input: &str, width: usize, height: usize) -> Result<Vec<u8>> {
    let mut child = Command::new("ffmpeg")
        .arg("-v")
        .arg("error")
        .arg("-nostdin")
        .arg("-i")
        .arg(input)
        .arg("-map")
        .arg("0:v:0")
        .arg("-f")
        .arg("rawvideo")
        .arg("-pix_fmt")
        .arg("rgb24")
        .arg("-vsync")
        .arg("0")
        .arg("-")
        .stdout(Stdio::piped())
        .spawn()
        .with_context(|| "failed to spawn ffmpeg")?;
    let mut out = Vec::new();
    child
        .stdout
        .as_mut()
        .ok_or_else(|| anyhow!("missing ffmpeg stdout pipe"))?
        .read_to_end(&mut out)
        .with_context(|| "failed to read ffmpeg stdout")?;
    let status = child.wait().with_context(|| "failed to wait for ffmpeg")?;
    if !status.success() {
        return Err(anyhow!("ffmpeg failed to decode video"));
    }
    let frame_size = width
        .checked_mul(height).ok_or_else(|| anyhow!("size overflow"))?
        .checked_mul(3).ok_or_else(|| anyhow!("size overflow"))?;
    let remainder = out.len() % frame_size;
    if remainder != 0 {
        // Drop any trailing partial frame
        let new_len = out.len() - remainder;
        out.truncate(new_len);
    }
    Ok(out)
}

fn load_video_impl(py: Python<'_>, input: &str) -> PyResult<(Py<PyAny>, f64, f64, usize, usize, usize)> {
    let meta = ffprobe_meta(input).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let bytes = ffmpeg_decode_rgb(input, meta.width, meta.height)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let frame_size = meta.width * meta.height * 3;
    if frame_size == 0 {
        return Err(PyRuntimeError::new_err("invalid frame size"));
    }
    let num_frames = bytes.len() / frame_size;
    let shape: (usize, usize, usize, usize) = (num_frames, meta.height, meta.width, 3usize);
    let nd: Array4<u8> = Array4::from_shape_vec(shape, bytes)
        .map_err(|e| PyRuntimeError::new_err(format!("failed to build ndarray: {}", e)))?;
    let arr = nd.into_pyarray(py);
    let obj: Py<PyAny> = arr.unbind().into();
    Ok((obj, meta.fps, meta.duration, meta.width, meta.height, num_frames))
}

#[pyfunction]
fn load_from_path(py: Python<'_>, path: &str) -> PyResult<(Py<PyAny>, f64, f64, usize, usize, usize)> {
    load_video_impl(py, path)
}

#[pyfunction]
fn load_from_url(py: Python<'_>, url: &str) -> PyResult<(Py<PyAny>, f64, f64, usize, usize, usize)> {
    load_video_impl(py, url)
}

#[pymodule]
fn video_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_from_path, m)?)?;
    m.add_function(wrap_pyfunction!(load_from_url, m)?)?;
    Ok(())
}
