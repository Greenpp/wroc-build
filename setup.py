from setuptools import find_packages, setup

setup(
    name='wroc-build',
    description='Building footprint segmentation in Wrocław',
    version='0.1.0',
    url='https://github.com/Greenpp/wroc-build',
    author='Jakub Ciszek',
    packages=find_packages(),
    package_data={'wroclaw_building_footprint': ['model/seg_model.pt']},
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'Pillow',
        'opencv-python',
        'segmentation-models-pytorch',
        'OWSLib',
        'matplotlib',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
