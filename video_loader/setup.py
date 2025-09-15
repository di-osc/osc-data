from setuptools import setup
from setuptools_rust import RustExtension, Binding


setup(
    name="video_loader",
    version="0.1.0",
    packages=["video"],
    package_dir={"": "src"},
    rust_extensions=[
        RustExtension(
            "video_rs",
            path="rust/Cargo.toml",
            binding=Binding.PyO3,
            debug=False,
        )
    ],
    install_requires=["numpy>=1.24"],
    zip_safe=False,
)

