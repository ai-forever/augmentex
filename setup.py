from setuptools import setup, find_packages
from glob import glob

setup(
    name="augmentex",
    version="1.3.1",
    author="Mark Baushenko and Alexandr Abramov",
    author_email="m.baushenko@gmail.com",
    description="Augmentex — a library for augmenting texts with errors",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/ai-forever/augmentex",
    packages=find_packages(),
    data_files=[("static_data", glob("augmentex/static_data/**/*.json", recursive=True))],
    include_package_data=True,
    classifiers=[
        "Natural Language :: English",
        "Natural Language :: Russian",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Editors :: Text Processing",
    ],
    python_requires=">=3.7.0",
    install_requires=["numpy>=1.21", "python-Levenshtein>=0.22.0"],
    keywords="augmentex errors typos nlp augmentation",
    zip_safe=False,
)