# Performance comparison between loading individual `.npy` files and using a HDF dataset in Pytorch

We are interested in fast loading of compressed datasets in Pytorch.

<!-- *Dated: June 2019* -->

## Data

10k data points of time-series measurements &mdash; basically float arrays of length ~100k. So when using `.npy` files, we have a folder with 10k files. When using HDF, we have a *HDF dataset* of size 10k x 100k. The data takes ~6.9GB when uncompressed and ~1.2GB when compressed (both HDF and flat files).

## Results

![comparison](https://github.com/chanshing/pytorch-h5py-comparison/blob/master/comparison.png)

- Machine: Core i7 8th gen, 16GB RAM.
- Compression: `blosc:lz4hc`, level 9, no shuffling. Raw size: **6.9GB**. Compressed size: **1.2GB**.
- Compression actually improved loading speeds slightly, presumably because we alleviate I/O operations in exchange for CPU operations to do the decompression.

## Notes

- We use `h5py`, a minimal Python package for interfacing with HDF5.
Another alternative is [PyTables](https://www.pytables.org/usersguide/introduction.html) which provides additional functionalities. Pandas also supports HDF via its `pandas.HDFStore` but data has to be in `pandas.DataFrame`'s.
- The compression that worked best for our dataset was [Blosc](http://blosc.org/pages/blosc-in-depth/), specifically `blosc:lz4hc` with compression level 9 and no [shuffling](http://python-blosc.blosc.org/tutorial.html#using-different-filters).
- Blosc compression in Python is provided via `python-blosc` module (http://python-blosc.blosc.org/intro.html#what-is-python-blosc), or is semi-integrated in `h5py` (see next point). We use `python-blosc` to compress the individual `.npy` files, and `h5py` integrated functionality to compress the HDF dataset (all with the same compressor parameters).
- `h5py` does not come with `blosc` compression out-of-the-box, but an easy solution is to `import tables` (PyTables does come with `blosc`) and `h5py` will automatically be able to find it. However we have to use a hacky way to interface with it &mdash; see [this comment](https://github.com/h5py/h5py/issues/611#issuecomment-353694301) or `create_data.py` of this repo for an example.
- `h5py` automatically enables *chunking* when compression is enabled. Since we will be retrieving entire data points, it helps to manually set chunks of size equal to the data point size. Chunking can have a huge effect in read performance,
see [this answer](https://stackoverflow.com/a/27713489/3250500).
- There seems to be an issue with multiprocessing in `h5py`. To work well with Pytorch `Dataloader` multiprocessing (`num_workers>1`), be sure to open `h5py.File` inside `__getitem__`. See [this comment](https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16).
- `python-blosc` can directly compress/decompress Numpy arrays with its `blosc.pack_array()` and `blosc.unpack_array()` methods, but it is supposed to be slow. They show an [alternative](http://python-blosc.blosc.org/tutorial.html#packaging-numpy-arrays) using pointers &mdash; note that with this method we need to know the data size to allocate beforehand which is cumbersome.

## Miscellaneous
- To clear cache for testing purposes,
    - Linux: `sudo sh -c 'free && sync && echo 3 > /proc/sys/vm/drop_caches && free'`.
    - Windows: use [RAMMap](https://docs.microsoft.com/en-us/sysinternals/downloads/rammap).
- To check HDF5 installation, `h5cc --showconfig`
- [This nice benchmark](http://alimanfoo.github.io/2016/09/21/genotype-compression-benchmark.html) of compressors performed on genotype datasets.
- [Fine-tuning](http://python-blosc.blosc.org/tutorial.html#fine-tuning-compression-parameters) options for Blosc.
- Incrementally build HDF datasets: https://stackoverflow.com/a/25656175/3250500
- A [nice summary](https://discuss.pytorch.org/t/how-to-prefetch-data-when-processing-with-gpu/548/19) of performance issues we may encounter.
