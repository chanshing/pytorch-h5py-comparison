import argparse

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgumentParser, self).__init__()
        # dataset
        self.add_argument('--load', type=str, default=None)
        self.add_argument('--numpy', action='store_true')
        self.add_argument('--h5py', action='store_true')
        self.add_argument('--seed', type=int, default=42)
        self.add_argument('--dataname', type=str, default=None)
        self.add_argument('--nsample', type=int, default=100)
        self.add_argument('--ndim', type=int, default=128)
        self.add_argument('--prefix', type=str, default='x')
        self.add_argument('--compress', action='store_true')
        self.add_argument('--codec', type=str, default='lz4hc')
        self.add_argument('--shuffle', type=int, default=0)
        self.add_argument('--clevel', type=int, default=9)
        self.add_argument('--norowchunk', action='store_true')
        # loading
        self.add_argument('--batch_size', type=int, default=8)
        self.add_argument('--num_workers', type=int, default=0)
