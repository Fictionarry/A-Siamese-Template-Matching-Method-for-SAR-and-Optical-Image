import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, template_dir: str, search_dir: str, loc_csv:str,  scale: float = 1.0, search_suffix: str = ''):
        self.template_dir = Path(template_dir)
        self.search_dir = Path(search_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.search_suffix = search_suffix

        if loc_csv:
            self.loc_df = pd.read_csv(loc_csv)

        self.ids = [splitext(file)[0] for file in listdir(template_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {template_dir}, make sure you put your template there')

        self.ids = np.int16(self.ids)
        self.ids.sort()

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized img would have no pixel'
        pil_img = pil_img.resize((newW, newH),Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2:
            img_ndarray = img_ndarray[np.newaxis, ...]
        else:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = str(self.ids[idx])
        search_file = list(self.search_dir.glob(name + self.search_suffix + '.*'))
        template_file = list(self.template_dir.glob(name + '.*'))

        assert len(search_file) == 1, f'Either no search or multiple search found for the ID {name}: {search_file}'
        assert len(template_file) == 1, f'Either no template or multiple template found for the ID {name}: {template_file}'
        search = self.load(search_file[0])
        template = self.load(template_file[0])


        template = self.preprocess(template, self.scale)
        search = self.preprocess(search, self.scale)

        df = self.loc_df[self.loc_df.name==self.ids[idx]]

        loc = [float(df.lt_x), float(df.lt_y), float(df.rb_x), float(df.rb_y)]

        return {
            'template': torch.as_tensor(template.copy()).float().contiguous(),
            'search': torch.as_tensor(search.copy()).float().contiguous(),
            'loc': torch.tensor(loc)
        }
