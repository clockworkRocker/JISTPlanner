# JIST

This is the implementation for [Joint Sampling and Trajectory Optimization Over Graphs for Online Motion Planning](https://arxiv.org/abs/2011.07171), Kalyan Vasudev Alwala and Mustafa Mukadam, IROS 2021.

JIST (JoInt Sampling and Trajectory optimization), is a unified approach that leverages the complementary strengths of sampling and optimization, and interleaves them both to tackle highly dynamic environments with long horizons that necessitate a fast online solution. See [project page](https://sites.google.com/view/jistplanner) for more details.


## Installation

- Install GTSAM and GPMP2:
  ```bash
  git clone --single-branch https://github.com/borglab/GTSAM.git
  mkdir GTSAM/build && cd GTSAM/build
  cmake -DGTSAM_BUILD_PYTHON=ON -DGTSAM_INSTALL_CYTHON_TOOLBOX=ON -DGTSAM_ALLOW_DEPRECATED_SINCE_V43=OFF ..
  make check # Run unit tests
  sudo make install # You only need sudo for installation into default location (/usr/local/lib)
  sudo make python-install
  ```

  ```
  git clone --single-branch https://github.com/borglab/GPMP2.git
  mkdir GPMP2/build && cd GPMP2/build
  cmake -DGPMP2_INSTALL_PYTHON_TOOLBOX=ON ..
  make check
  sudo make install
  make python-install
  ```

- Install OpenCV 4.2.0.32: `pip install opencv-python==4.2.0.32`


## Usage
To run a visual test of JIST run `testPlanner.py`

To use JIST in your projects use class `JISTPlanner` defined in module `modules.planner` and its `plan` method.

## Citation

If you use JIST in an academic context, please cite following publication.

```latex
@inproceedings{alwala2021joint,
  title={Joint sampling and trajectory optimization over graphs for online motion planning},
  author={Alwala, Kalyan Vasudev and Mukadam, Mustafa},
  booktitle={International Conference on Intelligent Robots and Systems (IROS)},
  year={2021}
}
```


License
-----

JIST is released under the BSD license, reproduced in [LICENSE](../../LICENSE).
