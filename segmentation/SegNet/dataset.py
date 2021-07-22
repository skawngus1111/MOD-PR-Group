import os
import random
import pickle

import torch
from torch.utils.data import Dataset

import numpy as np
from tqdm import trange
from pycocotools import mask
from pycocotools.coco import COCO
from PIL import Image, ImageOps, ImageFilter

class COCODataset(Dataset) :
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
                1, 64, 20, 63, 7, 72]
    NUM_CLASS = 21
    def __init__(self,
                 root_dir='/media/jhnam19960514/68334fe0-2b83-45d6-98e3-76904bf08127/home/namjuhyeon/Desktop/LAB/common material/Dataset Collection/COCO',
                 year='2014',
                 split='train',
                 transform=None,
                 base_size=520,
                 crop_size=480):
        super(COCODataset, self).__init__()

        self.root_dir = root_dir
        self.year     = year
        self.split    = split

        print(f'{self.split} dataset')
        ann_file = os.path.join(self.root_dir, self.year, 'annotations', 'instances_' + self.split + self.year + '.json')
        ids_file = os.path.join(self.root_dir, self.year, 'annotations', self.split + '_ids.mx')

        self.coco      = COCO(ann_file)
        self.coco_mask = mask

        if os.path.exists(ids_file) :
            with open(ids_file, 'rb') as f :
                self.ids = pickle.load(f)
        else :
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)

        self.transform = transform
        self.base_size = base_size
        self.crop_size = crop_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id       = self.ids[idx]
        img_metadata = self.coco.loadImgs(img_id)[0]
        path         = img_metadata['file_name']

        img        = Image.open(os.path.join(self.root_dir, self.year, self.split + self.year, path)).convert('RGB')
        cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        mask       = Image.fromarray(self._get_seg_mask(
            cocotarget, img_metadata['height'], img_metadata['width']
        ))

        # synchrosized transform
        if self.split == 'train' :
            img, mask = self._sync_transform(img, mask)
        elif self.split == 'val' :
            img, mask = self._val_sync_transform(img, mask)
        else :
            assert self.split == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)

        # general resize, normalize and toTensor
        if self.transform is not None :
            img = self.transform(img)

        return img, mask

    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, this will take a while." + \
              "But don't worry, it only run once for each split.")

        tbar = trange(len(ids))
        new_ids = []

        for i in tbar :
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._get_seg_mask(cocotarget, img_metadata['height'], img_metadata['width'])

            # more than 1k pixels
            if (mask > 0).sum() > 1000 :
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'.format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))

        with open(ids_file, 'wb') as f:
            pickle.dump(new_ids, f)

        return new_ids

    def _get_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask

        for instance in target :
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']

            if cat in self.CAT_LIST :
                c = self.CAT_LIST.index(cat)
            else :
                continue

            if len(m.shape) < 3 :
                mask[:, :] += (mask == 0) * (m * c)
            else :
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)

        return mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)

        return img, mask

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)

        return img, mask

    def _img_transform(self, img):
        return torch.from_numpy(np.transpose(np.array(img), (2, 0, 1)).copy()).float()

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32'))
