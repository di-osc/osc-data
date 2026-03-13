use image::{DynamicImage, ImageBuffer, Rgb, Rgba, Luma, ImageFormat};
use numpy::{PyArray3, PyArray4, PyReadonlyArray3, PyReadonlyArray4};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::io::Cursor;

/// Convert image data to RGB format.
/// Args:
///     data: numpy array of shape (H, W, C) uint8
///     from_format: source format string ("RGB", "RGBA", "L")
/// Returns:
///     numpy array of shape (H, W, 3) uint8
#[pyfunction]
fn convert_to_rgb<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<u8>,
    from_format: &str,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let arr = data.as_array();
    let shape = arr.shape();
    if shape.len() != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input array must be 3D (H, W, C)"
        ));
    }
    let height = shape[0];
    let width = shape[1];
    let channels = shape[2];

    let rgb_data: Vec<u8> = match from_format.to_uppercase().as_str() {
        "RGB" => {
            if channels != 3 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "RGB format requires 3 channels"
                ));
            }
            arr.as_standard_layout().as_slice().unwrap().to_vec()
        }
        "RGBA" => {
            if channels != 4 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "RGBA format requires 4 channels"
                ));
            }
            let slice = arr.as_standard_layout();
            let bytes = slice.as_slice().unwrap();
            let mut rgb = Vec::with_capacity(height * width * 3);
            for i in 0..(height * width) {
                rgb.push(bytes[i * 4]);
                rgb.push(bytes[i * 4 + 1]);
                rgb.push(bytes[i * 4 + 2]);
            }
            rgb
        }
        "L" | "GRAYSCALE" => {
            if channels != 1 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Grayscale format requires 1 channel"
                ));
            }
            let slice = arr.as_standard_layout();
            let bytes = slice.as_slice().unwrap();
            let mut rgb = Vec::with_capacity(height * width * 3);
            for &gray in bytes {
                rgb.push(gray);
                rgb.push(gray);
                rgb.push(gray);
            }
            rgb
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unsupported format: {}", from_format)
            ));
        }
    };

    let out_array = PyArray3::from_owned_array(
        py,
        ndarray::Array::from_shape_vec([height, width, 3], rgb_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to create array: {}", e)
            ))?
    );
    Ok(out_array)
}

/// Resize image data using high-quality Lanczos3 filter.
/// Args:
///     data: numpy array of shape (H, W, C) uint8
///     target_height: target height
///     target_width: target width
/// Returns:
///     numpy array of shape (target_height, target_width, C) uint8
#[pyfunction]
fn resize_image<'py>(
    _py: Python<'py>,
    data: PyReadonlyArray3<u8>,
    target_height: u32,
    target_width: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let arr = data.as_array();
    let shape = arr.shape();
    if shape.len() != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input array must be 3D (H, W, C)"
        ));
    }
    let height = shape[0] as u32;
    let width = shape[1] as u32;
    let channels = shape[2];

    let img = match channels {
        3 => {
            let slice = arr.as_standard_layout();
            let bytes = slice.as_slice().unwrap();
            DynamicImage::ImageRgb8(
                ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, bytes.to_vec())
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Failed to create RGB image buffer"
                    ))?
            )
        }
        4 => {
            let slice = arr.as_standard_layout();
            let bytes = slice.as_slice().unwrap();
            DynamicImage::ImageRgba8(
                ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, bytes.to_vec())
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Failed to create RGBA image buffer"
                    ))?
            )
        }
        1 => {
            let slice = arr.as_standard_layout();
            let bytes = slice.as_slice().unwrap();
            DynamicImage::ImageLuma8(
                ImageBuffer::<Luma<u8>, _>::from_raw(width, height, bytes.to_vec())
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Failed to create grayscale image buffer"
                    ))?
            )
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unsupported channel count: {}", channels)
            ));
        }
    };

    let resized = img.resize(target_width, target_height, image::imageops::FilterType::Lanczos3);

    let (out_w, out_h) = (resized.width() as usize, resized.height() as usize);
    let out_vec = match resized {
        DynamicImage::ImageRgb8(img) => img.into_raw(),
        DynamicImage::ImageRgba8(img) => img.into_raw(),
        DynamicImage::ImageLuma8(img) => img.into_raw(),
        _ => {
            // Convert to RGB8 if other format
            resized.to_rgb8().into_raw()
        }
    };

    let final_channels = match channels {
        1 => 1,
        3 => 3,
        4 => 4,
        _ => 3,
    };
    let out_array = PyArray3::from_owned_array(
        _py,
        ndarray::Array::from_shape_vec([out_h, out_w, final_channels], out_vec)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to create array: {}", e)
            ))?
    );
    Ok(out_array)
}

/// Crop image data.
/// Args:
///     data: numpy array of shape (H, W, C) uint8
///     x: x coordinate of top-left corner
///     y: y coordinate of top-left corner
///     width: crop width
///     height: crop height
/// Returns:
///     numpy array of shape (height, width, C) uint8
#[pyfunction]
fn crop_image<'py>(
    _py: Python<'py>,
    data: PyReadonlyArray3<u8>,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let arr = data.as_array();
    let shape = arr.shape();
    if shape.len() != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input array must be 3D (H, W, C)"
        ));
    }
    let img_height = shape[0] as u32;
    let img_width = shape[1] as u32;
    let channels = shape[2];

    if x + width > img_width || y + height > img_height {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Crop region exceeds image bounds"
        ));
    }

    let slice = arr.as_standard_layout();
    let bytes = slice.as_slice().unwrap();

    let mut cropped = Vec::with_capacity((height * width * channels as u32) as usize);
    for row in y..(y + height) {
        let row_start = ((row * img_width + x) * channels as u32) as usize;
        let row_end = row_start + (width * channels as u32) as usize;
        cropped.extend_from_slice(&bytes[row_start..row_end]);
    }

    let out_array = PyArray3::from_owned_array(
        _py,
        ndarray::Array::from_shape_vec([height as usize, width as usize, channels], cropped)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to create array: {}", e)
            ))?
    );
    Ok(out_array)
}

/// Encode image data to bytes in specified format.
/// Args:
///     data: numpy array of shape (H, W, C) uint8
///     format: output format ("png", "jpeg", "webp")
///     quality: quality for lossy formats (0-100), ignored for PNG
/// Returns:
///     bytes object containing encoded image
#[pyfunction]
fn encode_image<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<u8>,
    format: &str,
    _quality: Option<u8>,
) -> PyResult<Bound<'py, PyBytes>> {
    let arr = data.as_array();
    let shape = arr.shape();
    if shape.len() != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input array must be 3D (H, W, C)"
        ));
    }
    let height = shape[0] as u32;
    let width = shape[1] as u32;
    let channels = shape[2];

    let slice = arr.as_standard_layout();
    let bytes = slice.as_slice().unwrap();

    let img: DynamicImage = match channels {
        3 => DynamicImage::ImageRgb8(
            ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, bytes.to_vec())
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Failed to create RGB image buffer"
                ))?
        ),
        4 => DynamicImage::ImageRgba8(
            ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, bytes.to_vec())
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Failed to create RGBA image buffer"
                ))?
        ),
        1 => DynamicImage::ImageLuma8(
            ImageBuffer::<Luma<u8>, _>::from_raw(width, height, bytes.to_vec())
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Failed to create grayscale image buffer"
                ))?
        ),
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unsupported channel count: {}", channels)
            ));
        }
    };

    let mut buffer = Cursor::new(Vec::new());

    let img_format = match format.to_lowercase().as_str() {
        "png" => ImageFormat::Png,
        "jpeg" | "jpg" => ImageFormat::Jpeg,
        "webp" => ImageFormat::WebP,
        "bmp" => ImageFormat::Bmp,
        "gif" => ImageFormat::Gif,
        "tiff" => ImageFormat::Tiff,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unsupported format: {}", format)
            ));
        }
    };

    if img_format == ImageFormat::Jpeg || img_format == ImageFormat::WebP {
        // Convert to RGB8 for lossy formats if necessary
        let rgb_img = img.to_rgb8();
        rgb_img.write_to(&mut buffer, img_format)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to encode image: {}", e)
            ))?;
    } else {
        img.write_to(&mut buffer, img_format)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to encode image: {}", e)
            ))?;
    }

    Ok(PyBytes::new(py, &buffer.into_inner()))
}

/// Decode bytes to image data.
/// Args:
///     data: bytes object containing image data
/// Returns:
///     tuple of (numpy array (H, W, C) uint8, format string)
#[pyfunction]
fn decode_image<'py>(
    _py: Python<'py>,
    data: &[u8],
) -> PyResult<(Bound<'py, PyArray3<u8>>, String)> {
    let img = image::load_from_memory(data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to decode image: {}", e)
        ))?;

    let (width, height) = (img.width(), img.height());
    let format_str = format!("{:?}", img.color());

    let (channels, raw_data): (usize, Vec<u8>) = match img {
        DynamicImage::ImageRgb8(img) => (3, img.into_raw()),
        DynamicImage::ImageRgba8(img) => (4, img.into_raw()),
        DynamicImage::ImageLuma8(img) => (1, img.into_raw()),
        DynamicImage::ImageLumaA8(img) => {
            // Convert LumaA to RGBA
            let rgba: Vec<u8> = img.pixels()
                .flat_map(|p| vec![p[0], p[0], p[0], p[1]])
                .collect();
            (4, rgba)
        }
        _ => {
            // Convert other formats to RGB8
            (3, img.to_rgb8().into_raw())
        }
    };

    let out_array = PyArray3::from_owned_array(
        _py,
        ndarray::Array::from_shape_vec([height as usize, width as usize, channels], raw_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to create array: {}", e)
            ))?
    );
    Ok((out_array, format_str))
}

/// Get image info without full decoding.
/// Args:
///     data: bytes object containing image data
/// Returns:
///     tuple of (width, height, format_string)
#[pyfunction]
fn get_image_info(_py: Python<'_>, data: &[u8]) -> PyResult<(u32, u32, String)> {
    let reader = image::ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to read image: {}", e)
        ))?;

    let format = reader.format();
    let dimensions = reader.into_dimensions()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to get image dimensions: {}", e)
        ))?;

    let format_str = format.map(|f| format!("{:?}", f)).unwrap_or_else(|| "Unknown".to_string());
    Ok((dimensions.0, dimensions.1, format_str))
}

/// Batch resize multiple images.
/// Args:
///     data: numpy array of shape (N, H, W, C) uint8
///     target_height: target height
///     target_width: target width
/// Returns:
///     numpy array of shape (N, target_height, target_width, C) uint8
#[pyfunction]
fn batch_resize<'py>(
    _py: Python<'py>,
    data: PyReadonlyArray4<u8>,
    target_height: u32,
    target_width: u32,
) -> PyResult<Bound<'py, PyArray4<u8>>> {
    let arr = data.as_array();
    let shape = arr.shape();
    if shape.len() != 4 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input array must be 4D (N, H, W, C)"
        ));
    }
    let batch_size = shape[0];
    let channels = shape[3];
    let out_size = (batch_size * target_height as usize * target_width as usize * channels) as usize;
    let mut output = Vec::with_capacity(out_size);

    for i in 0..batch_size {
        let img_view = arr.slice(ndarray::s![i, .., .., ..]);
        let slice = img_view.as_standard_layout();
        let bytes = slice.as_slice().unwrap();

        let height = shape[1] as u32;
        let width = shape[2] as u32;

        let img = match channels {
            3 => DynamicImage::ImageRgb8(
                ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, bytes.to_vec())
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Failed to create RGB image buffer"
                    ))?
            ),
            4 => DynamicImage::ImageRgba8(
                ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, bytes.to_vec())
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Failed to create RGBA image buffer"
                    ))?
            ),
            1 => DynamicImage::ImageLuma8(
                ImageBuffer::<Luma<u8>, _>::from_raw(width, height, bytes.to_vec())
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Failed to create grayscale image buffer"
                    ))?
            ),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Unsupported channel count: {}", channels)
                ));
            }
        };

        let resized = img.resize(target_width, target_height, image::imageops::FilterType::Lanczos3);
        let raw = match resized {
            DynamicImage::ImageRgb8(img) => img.into_raw(),
            DynamicImage::ImageRgba8(img) => img.into_raw(),
            DynamicImage::ImageLuma8(img) => img.into_raw(),
            _ => resized.to_rgb8().into_raw(),
        };
        output.extend_from_slice(&raw);
    }

    let out_array = PyArray4::from_owned_array(
        _py,
        ndarray::Array::from_shape_vec(
            [batch_size, target_height as usize, target_width as usize, channels],
            output
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Failed to create output array: {}", e)
        ))?
    );
    Ok(out_array)
}

/// Normalize image from uint8 [0, 255] to float32, with optional mean/std per channel.
///
/// Args:
///     data: numpy array of shape (H, W, C) uint8
///     mean: per-channel mean, e.g. [0.485, 0.456, 0.406] for ImageNet RGB
///     std: per-channel std, e.g. [0.229, 0.224, 0.225] for ImageNet RGB
///
/// Returns:
///     numpy array of shape (H, W, C) float32, computed as (pixel/255.0 - mean) / std
#[pyfunction]
fn normalize_image<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<u8>,
    mean: Vec<f32>,
    std: Vec<f32>,
) -> PyResult<Bound<'py, numpy::PyArray3<f32>>> {
    let arr = data.as_array();
    let shape = arr.shape();
    let height = shape[0];
    let width = shape[1];
    let channels = shape[2];

    if mean.len() != channels || std.len() != channels {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "mean/std length ({}/{}) must match channel count ({})",
            mean.len(),
            std.len(),
            channels
        )));
    }

    for (i, &s) in std.iter().enumerate() {
        if s == 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "std[{}] is zero, division by zero",
                i
            )));
        }
    }

    let slice = arr.as_standard_layout();
    let bytes = slice.as_slice().unwrap();
    let total = height * width * channels;
    let mut output = Vec::with_capacity(total);

    for pixel in bytes.chunks_exact(channels) {
        for c in 0..channels {
            output.push((pixel[c] as f32 / 255.0 - mean[c]) / std[c]);
        }
    }

    let out_array = numpy::PyArray3::from_owned_array(
        py,
        ndarray::Array::from_shape_vec([height, width, channels], output).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to create array: {}",
                e
            ))
        })?,
    );
    Ok(out_array)
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(convert_to_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(resize_image, m)?)?;
    m.add_function(wrap_pyfunction!(crop_image, m)?)?;
    m.add_function(wrap_pyfunction!(encode_image, m)?)?;
    m.add_function(wrap_pyfunction!(decode_image, m)?)?;
    m.add_function(wrap_pyfunction!(get_image_info, m)?)?;
    m.add_function(wrap_pyfunction!(batch_resize, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_image, m)?)?;
    Ok(())
}
