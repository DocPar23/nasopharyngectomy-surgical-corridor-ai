from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="npc-surgical-planning",
    version="1.0.0",
    author="Dr. Parnini Goswami",
    author_email="your.email@example.com",
    description="AI-driven surgical risk zone mapping for Nasopharyngeal Carcinoma",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/npc-surgical-planning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=[
        "nibabel>=5.2.0",
        "numpy>=1.26.4",
        "scipy>=1.11.4",
        "matplotlib>=3.8.2",
        "pandas>=2.1.4",
        "scikit-image>=0.22.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.3", "black>=23.12.0", "flake8>=7.0.0"],
        "deep_learning": ["torch>=2.1.0", "monai>=1.3.0"],
    },
)
```

---

## **8. LICENSE (MIT License)**
```
MIT License

Copyright (c) 2024 Dr. Parnini Goswami, Dr. Satish Nair, Dr. Vadhiraja

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
