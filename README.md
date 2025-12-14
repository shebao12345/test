<p align="center">
  <h1 align="center">AAAAA: BBB</h1>
  <p align="center">
  YYY*, ZZZ*, SC, VF



<p align="center">
  <a href="">
    <img src="./save/1746717535947.png" alt="Logo" width="100%">
  </a>
</p>
Pixel-aligned feed-forward 3DGS methods suffer from two primary limitations: 1) 2D feature matching struggles to effectively resolve the multi-view alignment problem, and 2) the Gaussian density is constrained and cannot be adaptively controlled according to scene complexity. We propose VolSplat, a method that directly regresses Gaussians from 3D features based on a voxel-aligned prediction strategy. This approach achieves adaptive control over scene complexity and resolves the multi-view alignment challenge.

## Method
<p align="center">
  <a href="">
    <img src="./save/QQ_1758607195449.png" alt="Logo" width="100%">
  </a>
</p>
<strong>Overview of VolSplat</strong>. Given multi-view images as input, we first extract 2D features for each image using a Transformer-based network and construct per-view cost volumes with plane sweeping. Depth Prediction Module then estimates a depth map for each view, which is used to unproject the 2D features into 3D space to form a voxel feature grid. Subsequently, we employ a sparse 3D decoder to refine these features in 3D space and predict the parameters of a 3D Gaussian for each occupied voxel. Finally, novel views are rendered from the predicted 3D Gaussians.


## TODOs
- [√] Release Code.
- [ ] Release Model Checkpoints.

## Citation
If you find our work useful for your research, please consider citing us:

```bibtex
@article{XXX,
  title={VVVV},
  author={WWWW},
  journal={arXiv preprint arXiv:114514},
  year={2025}
}
```
## Contact
If you have any questions, please create an issue on this repository or contact at wangweijie@zju.edu.cn.

## Acknowledgements

This project is developed with [vggt](https://github.com/facebookresearch/114514). We thank the original authors for their excellent work.
