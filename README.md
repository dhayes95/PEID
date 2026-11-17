# PEID
Provides various functions that can attach to the provided tensor-train cross function, as well as tntorch.cross().

## Files

- src/tt_functions.py : Base TTACA code that will perform the construction and construct PEID cores.
- src/core_construct.py : Standalone oversampling PEID codes.
- examples/examples.py : Examples using TTACA + PEID and TnTorch + PEID

## Dependencies

- NumPy
- SciPy
- PyTorch - optional and only need for TnTorch example
- TnTorch - optional and only need for TnTorch example
- mpi4py - optional, only need for MPI methods

## Examples

To run the code, download tt_functions.py, core_construct.py, and examples.py to the same folder, then run examples.py with a choice of setup.

Available tensors: Hilbert, Matern, ThinPlateSpline, 2d2v, 3d3v

## Construction options

- None: TTACA with no oversampling
- Par: PEID-Par
- Seq: PEID-Seq
- Par2: PEID-Par2
- Seq2: PEID-Seq2
- R-Par: PEID-R using PEID-Par left to right and right to left sweeps
- R-Seq: PEID-R using PEID-Seq left to right and right to left sweeps
- Par-P: MPI based PEID-Par
- Par2-P: MPI based PEID-Par2

`adap_toler=None` retains the ranks determined by the TTACA index sets.
Set it to a numerical tolerance to enable adaptive rank selection.
The R methods always round back to the original ranks.

MPI methods require all processes to call the function, with at least
one process per tensor dimension. Cores are returned on rank 0.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{hayes2026efficient,
  title={An Efficient and Robust Projection Enhanced Interpolation Based Tensor Train Decomposition},
  author={Hayes, Daniel and Qiu, Jing-Mei and Shi, Tianyi},
  journal={arXiv preprint arXiv:2602.07653},
  year={2026}
}
```

[Read the paper](https://arxiv.org/abs/2602.07653).

