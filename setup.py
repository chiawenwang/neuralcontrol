from setuptools import setup
import glob
import os

# The compiled extension is produced by CMake (see README); pick it up if present.
so_files = glob.glob(os.path.join("nn_der", "nn_der*.so"))
package_data = [os.path.basename(p) for p in so_files] if so_files else []

setup(
    name="nn_der",
    version="0.1",
    packages=["nn_der"],
    package_dir={"nn_der": "nn_der"},
    package_data={"nn_der": package_data},
)
