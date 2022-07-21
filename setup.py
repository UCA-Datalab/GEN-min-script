from setuptools import setup, find_packages

setup(
    name="gen",
    version="0.1.1",
    packages=find_packages(),
    python_requires=">=3.9",
    setup_requires=["setuptools_scm"],
    install_requires=[
        "pandas>=1.4.1",
        "numpy>=1.22.2",
        "torch>=1.10.2",
        "torchvision>=0.11.3",
        "transformers>=4.16.2",
        "Pillow>=8.2.0",
        "easyocr>=1.4.1",
        "typer>=0.6.1",
        "pdf2image>=1.16.0"
    ],
    extras_require={
        "dev": ["pip-tools", "pytest>=6.2.3", "black>=20.8b1", "isort", "flake8",]
    },
)
