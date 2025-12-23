#    MIT License

#     Copyright (c) Microsoft Corporation.

#     Permission is hereby granted, free of charge, to any person obtaining a copy
#     of this software and associated documentation files (the "Software"), to deal
#     in the Software without restriction, including without limitation the rights
#     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#     copies of the Software, and to permit persons to whom the Software is
#     furnished to do so, subject to the following conditions:

#     The above copyright notice and this permission notice shall be included in all
#     copies or substantial portions of the Software.

#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#     SOFTWARE
"""TensorFlow Privacy library setup file for pip."""
from setuptools import find_packages
from setuptools import setup

setup(
    name='privGan',
    version='1.0',
    url='https://github.com/microsoft/privGAN',
    license='MIT',
    author ='Sumit Mukherjee, Nabajyoti Patowary',
    author_email='privgan@microsoft.com',
    description='Privacy protected GAN for image data (PyTorch)',
    long_description='PrivGAN implemented with PyTorch for privacy-preserving GAN training and evaluation.',
    long_description_content_type="text/markdown",
    install_requires=[
        'numpy>=1.16.2',
        'pandas>=0.25.3',
        'tqdm>=4.38.0',
        'scipy>=1.1.0',
        'matplotlib>=3.3.0',
        'scikit-learn>=0.20.3',
        'torch>=2.0.0',
        'torchvision>=0.15.0',
    ],
    packages=find_packages())