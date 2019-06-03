import os
import arguments
import random
import numpy
import blosc
import h5py
import tables  # required for blosc compression

def ndarray_to_flatnpy(data, dataname, prefix='x',
    compression=False, clevel=9, shuffle=1, codec='blosclz'):
    shuffle = (blosc.NOSHUFFLE, blosc.SHUFFLE, blosc.BITSHUFFLE)[shuffle]
    if compression:
        for i, d in enumerate(data):
            fname = '{}/{}_{}.blosc'.format(dataname, prefix, i)
            d = blosc.compress_ptr(d.__array_interface__['data'][0], d.size, d.dtype.itemsize, clevel=clevel, shuffle=shuffle, cname=codec)
            with open(fname, 'wb') as f:
                f.write(d)
    else:
        for i, d in enumerate(data):
            fname = '{}/{}_{}.npy'.format(dataname, prefix, i)
            numpy.save(fname, d)

def blosc_opts(clevel=9, shuffle=1, codec='blosclz'):
    codec = ['blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib', 'zstd'].index(codec)
    BLOSC=32001  # see: https://www.hdfgroup.org/services/filters.html
    args = {
        'compression': BLOSC,
        'compression_opts': (0, 0, 0, 0, clevel, shuffle, codec)
    }
    args['shuffle'] = False
    return args

def ndarray_to_hdf5(data, dataname, prefix='x',
    compression=False, clevel=9, shuffle=1, codec='blosclz', norowchunk=False):
    if compression:
        chunks = (1, *data.shape[1:]) if not norowchunk else True  # chunks=True is autochunk
        with h5py.File(dataname+'.h5', 'w-') as f:
            f.create_dataset(prefix, data=data, **blosc_opts(clevel, shuffle, codec), chunks=chunks)
    else:
        with h5py.File(dataname+'.h5', 'w-') as f:
            f.create_dataset(prefix, data=data)


if __name__ == '__main__':
    parser = arguments.ArgumentParser()
    args = parser.parse_args()

    random.seed(args.seed)
    numpy.random.seed(args.seed)

    if args.load is not None:
        data = numpy.load(args.load)
        args.nsample, args.ndim = data.shape
    else:
        print("generating fake data")
        data = numpy.random.randn(args.nsample, args.ndim)  # note: random data cannot be compressed!

    if args.dataname is None:
        args.dataname = '{}x{}'.format(args.nsample, args.ndim)
        if args.compress:
            args.dataname += '_C'
    print('Dataset name:', args.dataname)

    if args.numpy:
        print("creating npy folder")
        os.system('mkdir -p {}'.format(args.dataname))
        ndarray_to_flatnpy(data, args.dataname, args.prefix, args.compress, args.clevel, args.shuffle, args.codec)
    if args.h5py:
        print("creating h5 file")
        if args.norowchunk: args.dataname += '_NRC'
        ndarray_to_hdf5(data, args.dataname, args.prefix, args.compress, args.clevel, args.shuffle, args.codec, args.norowchunk)
