
"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import scipy.io
from scipy.io import loadmat
import h5py

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------
# Abstract base class for datasets.

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        cache       = False,    # Cache images in CPU memory?
    ):
        print(cache)
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict() # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])
        # self._raw_idx = self._raw_idx[:max_size]

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        #assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        #scipy.io.savemat('thingy.mat', {'x': image})
        # pad_width = ((0, 0), (32, 32), (32, 32))
        # x = np.pad(image.copy(), pad_width, mode='constant', constant_values=0)
        #print(image.shape)
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

class ImageFolderDataset5(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        pad = 64,
        channels = 1, #duplicating the volume multiple times
        zeropadding = True,
        cache=False
    ):
        self._path = path
        self.zeropadding = zeropadding
        self._zipfile = None
        self.pad = pad
        self.channels = channels
        self.width = resolution - 2*pad
        #print(self.alldata.shape) #should be N by 256 by 256 by 256

        name = 'what the hell is this'
        self.directories = os.listdir(path)
        for npfile in self.directories:
            assert npfile[-3:] == 'npy'
        self.images = len(self.directories)
        raw_shape = (self.images, channels, resolution, resolution)
        print('raw shape: ', raw_shape)
        super().__init__(name=name, raw_shape=raw_shape, cache=cache)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        return NotImplementedError
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        pad_width = ((self.pad, self.pad), (self.pad, self.pad))
        dir = os.path.join(self._path, self.directories[raw_idx])
        x = np.load(dir)
        if self.zeropadding:
            slice = np.pad(x, pad_width, mode='constant', constant_values=0)
            slice = np.expand_dims(slice, axis=0) #1 256 256
            if self.channels==1:
                return slice
            return np.concatenate([slice]*self.channels, axis=0)
        else:
            array = np.expand_dims(array, axis=0) #1 256 256 256
            #print('element size, ', array.shape)
            if self.channels==1:
                return array
            return np.concatenate([array]*self.channels, axis=0)


    def _load_raw_labels(self):
        return NotImplementedError

class ImageFolderDataset4(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        pad = 64,
        channels = 1, #duplicating the volume multiple times
        zeropadding = True,
        cache=False
    ):
        self._path = path
        self.zeropadding = zeropadding
        self._zipfile = None
        self.pad = pad
        self.channels = channels
        self.width = resolution - 2*pad
        #print(self.alldata.shape) #should be N by 256 by 256 by 256

        name = 'what the hell is this'
        self.directories = os.listdir(path)
        for matfile in self.directories:
            assert matfile[:3] == 'CT_' or matfile[:4] == 'data'
        self.volumes = len(self.directories)
        raw_shape = (self.volumes*256, channels, resolution, resolution)
        print('raw shape: ', raw_shape)
        super().__init__(name=name, raw_shape=raw_shape, cache=cache)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        return NotImplementedError
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        pad_width = ((self.pad, self.pad), (self.pad, self.pad))
        dir = os.path.join(self._path, self.directories[raw_idx//256])
        mat_data = loadmat(dir)
        array = mat_data['ImData_resample_resize']
        toret = None
        if self.zeropadding:
            imindex = raw_idx%256
            slice = np.squeeze(array[imindex, :, :])
            slice = np.pad(slice, pad_width, mode='constant', constant_values=0)
            slice = np.expand_dims(slice, axis=0) #1 256 256
            if self.channels==1:
                return slice
            return np.concatenate([slice]*self.channels, axis=0)
        else:
            array = np.expand_dims(array, axis=0) #1 256 256 256
            #print('element size, ', array.shape)
            if self.channels==1:
                return array
            return np.concatenate([array]*self.channels, axis=0)


    def _load_raw_labels(self):
        return NotImplementedError

class ImageFolderDataset3(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        pad = 64,
        bigdata = False, #if this, read from the folder directly
        channels = 3,
        imsize = 256,
        zlast = False
    ):
        self.channels = channels
        self._path = path
        self._zipfile = None
        self.bigdata = bigdata
        if zlast:
            # x = h5py.File(path, 'r')
            # x = x['allim']
            # x = np.transpose(x, (0,2,1))
            x = scipy.io.loadmat(path)['images']
            x = np.transpose(x, (2,0,1))
            self.alldata = np.expand_dims(x, 1)
        else:
            if not bigdata:
                print('image size: ', imsize)
                if imsize == 256:
                    x = scipy.io.loadmat(path)
                    x = x['bigdata']
                else:
                    x = h5py.File(path, 'r')
                    x = np.transpose(x['bigdata'], axes=(3,2,1,0))

                self.alldata = np.transpose(x, (1,2,3,0))
                self.alldata = np.reshape(self.alldata, (imsize, imsize, -1))
                self.alldata = np.transpose(self.alldata, (2,0,1))
                self.alldata = np.expand_dims(self.alldata, 1)
        self.pad = pad
        print(self.alldata.shape) #should be 2304 1 256 256

        if imsize > 256:
            self.alldata = self.alldata/2200
            self.alldata[self.alldata > 1] = 1
            print('max value of training data: ', self.alldata.max())

        name = 'ct1'
        if not bigdata:
            pad_width = ((0,0), (0,0), (self.pad, self.pad), (self.pad, self.pad))
            z = np.pad(self.alldata, pad_width, mode='constant', constant_values=0)
            z = np.concatenate([z]*channels, axis=1)
            raw_shape = z.shape
        # if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
        #     raise IOError('Image files do not match the specified resolution')
            super().__init__(name=name, raw_shape=raw_shape)
        else:
            return NotImplementedError

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        return NotImplementedError
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        pad_width = ((0,0), (self.pad, self.pad), (self.pad, self.pad))
        if not self.bigdata:
            image = self.alldata[raw_idx,:,:,:]
            image = np.pad(image.copy(), pad_width, mode='constant', constant_values=0)
            #print('padded image size ', image.shape)
            return np.concatenate([image]*self.channels, axis=0) #3 384 384
        else:
            return NotImplementedError

    def _load_raw_labels(self):
        return NotImplementedError

class ImageFolderDataset2(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        pad = 64,
        bigdata = False #if this, read from the folder directly
    ):
        self._path = path
        self._zipfile = None
        self.bigdata = bigdata
        if not bigdata:
            x = scipy.io.loadmat(path)
            self.alldata = np.expand_dims(np.transpose(x['y'], (2,0,1)), axis=1)
            self.alldata = np.repeat(self.alldata, 3, axis=1)
        self.pad = pad
        #print(self.alldata.shape) #should be N by 256 by 256 by 256

        name = 'spect'
        if not bigdata:
            pad_width = ((0,0), (0,0), (self.pad, self.pad), (self.pad, self.pad))
            z = np.pad(self.alldata, pad_width, mode='constant', constant_values=0)
            raw_shape = z.shape
        # if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
        #     raise IOError('Image files do not match the specified resolution')
            super().__init__(name=name, raw_shape=raw_shape)
        else:
            self.directories = os.listdir(path)
            for thing in self.directories:
                assert thing[:4] == 'LIDC'
            self.volumes = len(self.directories)
            raw_shape = (self.volumes, 1, 256+2*pad, 256+2*pad, 256+2*pad)
            super().__init__(name=name, raw_shape=raw_shape, pad=pad)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        return NotImplementedError
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        pad_width = ((0,0), (self.pad, self.pad), (self.pad, self.pad))
        if not self.bigdata:
            image = self.alldata[raw_idx,:,:,:]
            image = np.pad(image.copy(), pad_width, mode='constant', constant_values=0)
            #print('padded image size ', image.shape)
            return image #3 384 384
        else:
            return NotImplementedError
            #data = np.zeros((256, 256, 256))
            dir = os.path.join(self._path, self.directories[raw_idx])
            mat_data = loadmat(os.path.join(dir, 'thisdata.mat'))
            return mat_data['x']

    def _load_raw_labels(self):
        return NotImplementedError

#----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        use_pyspng      = True, # Use pyspng if available?
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._use_pyspng = use_pyspng
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        print('All data: ', raw_shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if self._use_pyspng and pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------
