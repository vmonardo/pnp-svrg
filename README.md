# Plug-and-Play with Variance Reduction Techniques

This repository contains all code needed to reproduce experiments in "Plug-and-Play Image Reconstruction Meets Stochastic Variance-Reduced Gradient Methods" [[PDF](https://users.ece.cmu.edu/~yuejiec/papers/PnPSVRG_icip2021.pdf)].

If you find this code useful, please cite our paper:
```
@inproceedings{monardo2021plug,
  title={Plug-and-Play Image Reconstruction Meets Stochastic Variance-Reduced Gradient Methods},
  author={Monardo, Vincent and Iyer, Abhiram and Donegan, Sean and de Graef, Marc and Chi, Yuejie},
  booktitle={ICIP 2021-2021 IEEE International Conference on Image Processing (ICIP)},
  year={2021},
  organization={IEEE}
}
```

## Requirements

- Python 3
- Required packages are list in ``requirements.txt``.


## Experiments
- Compressed Sensing for MRI: file ``create_paper_figures_csmri.ipynb``
- Deblurring and Super Resolution: file ``create_paper_figures_deblur.ipynb``
- Phase Retrieval: file ``create_paper_figures_pr.ipynb``