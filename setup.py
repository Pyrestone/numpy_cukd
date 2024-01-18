import subprocess
import os

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Copyright Marc Uecker (MIT License)

class CMakeBuild(build_ext):
    def run(self):
        source_dir=os.path.split(__file__)[0]
        build_temp = os.path.join(self.build_temp, 'cmake_build')
        os.makedirs(build_temp, exist_ok=True)

        self.announce(f"source_dir is: {source_dir}",3)
        self.announce(f"build_lib is: {self.build_lib}",3)
        self.announce(f"build_temp is: {build_temp}",3)

        # Run CMake in the temporary build directory
        subprocess.check_call(['cmake', source_dir, '-B' + build_temp])
        subprocess.check_call(['cmake', '--build', build_temp])

        # Ensure the 'mypackage' directory is created
        self.mkpath(self.build_lib)
        self.mkpath(os.path.join(self.build_lib,'numpy_cukd'))

        self.copy_file(os.path.join(build_temp, 'src/numpy_cukd.so'), os.path.join(self.build_lib,'numpy_cukd/numpy_cukd.so'))
        self.copy_tree(os.path.join(source_dir, 'src/numpy_cukd'), os.path.join(self.build_lib,'numpy_cukd'))

setup(
    name="numpy_cukd",
    version="0.0.1",
    author="Marc Uecker",
    author_email="marc.uecker@outlook.de",
    description="exposing cukd to numpy",
    long_description="exposing cukd to numpy - gotta go fast with nearest neighbor searches.",
    ext_modules=[Extension("numpy_cukd",[])],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.7",
)