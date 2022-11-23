import setuptools

setuptools.setup(
    name="zoobot-batch",
    version="0.1.0",
    author="Zooniverse Team",
    author_email="contact@zooniverse.org",
    description="Zoobot batch processing system",
    url="https://github.com/zooniverse/bajor/blob/main/azure/batch/README.md",
    license='Apache License 2.0',
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: Apache Software License'
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Environment :: GPU :: NVIDIA CUDA"
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.7",  # tf 2.8.0 requires Python 3.7 and above
    install_requires=[
        # 'zoobot[pytorch] >= 0.0.4', # the big cheese - bring in the zoobot!
        # 'zoobot[pytorch] @ git+https://github.com/mwalmsley/zoobot.git@dev',
        'zoobot[pytorch] @ git+https://github.com/camallen/zoobot.git@param-finetuning',
        'requests >= 2.28.1', # used to download prediction images from a remote URL
        'h5py >= 3.7.0' # used for prediction exports
    ]
)
