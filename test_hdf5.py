import time
import random
import arguments
import h5py
import tables  # required for blosc compression
import torch

from torch.utils import data

class H5Dataset(data.Dataset):
    def __init__(self, filename, prefix):
        if not filename.endswith('.h5'):
            filename = filename+'.h5'
        self.filename = filename
        self.prefix = prefix
        self.dataset = None
        with h5py.File(self.filename, 'r') as f:
            self.dataset_len = len(f[prefix])

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.filename, 'r')[self.prefix]
        return torch.from_numpy(self.dataset[index])

    def __len__(self):
        return self.dataset_len


if __name__ == '__main__':
    parser = arguments.ArgumentParser()
    parser.add_argument('dataname', type=str)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataloader = data.DataLoader(H5Dataset(args.dataname, args.prefix),
        batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    before = time.time()
    trash = 0.
    for x in dataloader:
        trash += x.sum()
    print(trash)
    print("took {:.4f} secs".format(time.time()-before))
