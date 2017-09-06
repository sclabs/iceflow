from setuptools import setup, find_packages

import versioneer

with open('README.md') as fobj:
    long_description = fobj.read()

setup(
    name='iceflow',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='tensorflow meta-framework',
    long_description=long_description,
    author='Thomas Gilgenast',
    url='https://github.com/sclabs/iceflow',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'iceflow = iceflow.iceflow:iceflow'
        ]
    },
    install_requires=[
        'tensorflow>=1.3.0',
        'dm-sonnet>=1.11'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
)
