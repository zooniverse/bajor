import setuptools

setuptools.setup(
    name="zoobot-batch",
    version="0.1.0",
    author="Zooniverse Team",
    author_email="contact@zooniverse.org",
    description="Zoobot batch processing system",
    url="https://github.com/zooniverse/bajor/blob/main/azure/batch/README.md",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Environment :: GPU :: NVIDIA CUDA"
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.7",  # tf 2.8.0 requires Python 3.7 and above
    install_requires=[
        'zoobot[pytorch] >= 0.0.4', # the big cheese - bring in the zoobot!
        'requests >= 2.28.1', # used to download prediction images from a remote URL
        'h5py >= 3.7.0' # used for prediction exports
    ]
)