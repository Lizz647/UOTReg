from setuptools import setup, find_packages

setup(
    name='UOTReg',
    version='0.1.0',  # Start with a dev version
    packages=find_packages(),  # Automatically finds 'UOTReg' folder as the package
    install_requires=[],  # Leave empty; handle dependencies in requirements.txt
    description='Trajectory inference method for temporal single-cell RNA-seq datasets',
    author='Binghao Yan',
    author_email='binghao.yan@pennmedicine.upenn.edu',
    url='https://github.com/Lizz647/UOTReg',
    license='MIT'  # Or whatever your LICENSE.md specifies
)