import os
import time
import arguments
import numpy
import torch
import blosc

from torch.utils import data

# blosc.set_nthreads(2)

class NumpyDataset(data.Dataset):
    def __init__(self, dataname, is_compressed=False, ndim=None):
        self.dataname = dataname
        self.is_compressed = is_compressed
        self.ndim = ndim
        if is_compressed:
            assert ndim is not None, 'Must pass ndim  explicitly to be able to decompress (--decomp_ndim)'
        self.filenames = [fname for fname in os.listdir(dataname) if os.path.isfile(os.path.join(dataname, fname))]
        self.dataset_len = len(self.filenames)

    def __getitem__(self, index):
        fname = os.path.join(self.dataname, self.filenames[index])
        if self.is_compressed:
            with open(fname, 'rb') as f:
                arr = numpy.empty(self.ndim, dtype=numpy.float64)
                blosc.decompress_ptr(f.read(), arr.__array_interface__['data'][0])
                return torch.from_numpy(arr)
        else:
            return torch.from_numpy(numpy.load(fname))

    def __len__(self):
        return self.dataset_len

if __name__ == '__main__':
    parser = arguments.ArgumentParser()
    parser.add_argument('dataname', type=str)
    parser.add_argument('--decomp_ndim', type=int, default=None)
    args = parser.parse_args()

    dataloader = data.DataLoader(NumpyDataset(args.dataname, args.compress, args.decomp_ndim),
        batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    before = time.time()
    trash = 0.
    for x in dataloader:
        trash += x.sum()
    print(trash)
    print("took {:.4f} secs".format(time.time()-before))
