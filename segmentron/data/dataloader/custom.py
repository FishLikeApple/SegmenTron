"""Prepare Custom dataset"""
import os
import torch
import numpy as np
import logging
import rasterio
import rasterio.mask
import fiona

from ...config import cfg

from PIL import Image
from .seg_data_base import SegmentationDataset

NUM_CLASS = cfg.DATASET.NUM_CLASS
TRAIN_PATH = cfg.DATASET.TRAIN_PATH
MASK_PATH = 'masks'
TEST_PATH = cfg.DATASET.TEST_PATH

class CustomSegmentation(SegmentationDataset):
    """Custom Semantic Segmentation Dataset.
    Parameters
    ----------
    root : string
        Path to folder. Default is 'circle-finder-marathon-challenge-train-data'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = CitySegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    NUM_CLASS = NUM_CLASS

    def __init__(self, train_root=TRAIN_PATH, test_root=TEST_PATH, split='train', mode=None, transform=None, **kwargs):
        super(CustomSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        if self.mode == "test":
            self.root = test_root
            assert os.path.exists(self.root)
            self.images = _get_test_dataset_items(self.root)
        else:
            self.root = train_root
            assert os.path.exists(self.root)
            _get_masks(self.root)
            self.images, self.mask_paths = _get_dataset_pairs(self.root, MASK_PATH, self.split)
            assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        self._key = np.array([0, 255])

    def _class_to_index(self, mask):
        # assert the value
        values = np.unique(mask)
        for value in values:
            assert (value in self._key)
        return mask//255

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._test_sync_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.mask_paths[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        #img = img / 255.0
        '''
        ay = (img).astype('uint8')
        im = Image.fromarray(ay)
        im.save("./image{}.jpeg".format(os.path.basename(self.images[index])))
        _ay = np.expand_dims(mask, 2)
        ay = (np.concatenate([_ay, _ay, _ay], 2)*255).astype('uint8')
        im = Image.fromarray(ay)
        im.save("./mask{}.jpeg".format(os.path.basename(self.images[index])))
        '''
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, os.path.basename(self.images[index])

    def _mask_transform(self, mask):
        target = self._class_to_index(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(target).astype('int32'))

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

    @property
    def classes(self):
        """Category names."""
        return ('circle')
    
def _get_mask(image_file, anno_file, output_mask_file):
    with fiona.open(anno_file, "r") as annotation_collection:
        annotations = [feature["geometry"] for feature in annotation_collection]
                    
    with rasterio.open(image_file) as src:
        out_image, out_transform = rasterio.mask.mask(src, annotations, all_touched=False, invert=True)
        out_meta = src.meta
        
    with rasterio.open(output_mask_file, "w", **out_meta) as dest:
        dest.write(out_image)

    with rasterio.open(output_mask_file) as src:
        out_image, out_transform = rasterio.mask.mask(src, annotations, all_touched=False, nodata=255, invert=False)
        out_meta = src.meta

        out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(output_mask_file, "w", **out_meta) as dest:
        dest.write(out_image)

def _get_masks(data_folder, output_path='masks/'):
    print("start getting masks...")
    for root, dirs, _ in os.walk(data_folder):
            for dir in dirs:
                foldername = os.path.basename(dir)
                image_path = os.path.join(data_folder, dir, foldername+'_PAN.tif')
                anno_path = os.path.join(data_folder, dir, foldername+'_anno.geojson')
                _get_mask(image_path, anno_path, output_path+foldername+'_mask.tif')
                print("mask "+foldername+" is done.")

def _get_dataset_pairs(data_folder, mask_folder, split='train', random_seed=6):
    print("start getting dataset pairs...")
    def get_path_pairs(data_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, dirs, _ in os.walk(data_folder):
            for dir in dirs:
                foldername = os.path.basename(dir)
                imgpath = os.path.join(data_folder, dir, foldername+'_PAN.tif')
                maskpath = os.path.join(mask_folder, foldername+'_mask.tif')
                if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                    print("pair "+foldername+" is done.")
                else:
                    print('cannot find the mask or image:', imgpath, maskpath)
        logging.info('Found {} images in the folder {}'.format(len(img_paths), data_folder))
        return img_paths, mask_paths
    
    img_paths, mask_paths = get_path_pairs(data_folder, mask_folder)
    np.random.seed(random_seed)
    np.random.shuffle(img_paths)
    np.random.seed(random_seed)
    np.random.shuffle(mask_paths)
    
    if split == 'train':
        return img_paths[:-len(img_paths)//10], mask_paths[:-len(img_paths)//10]
    elif split == 'val':
        return img_paths[-len(img_paths)//10:], mask_paths[-len(img_paths)//10:]
    assert split == 'trainval'
    return img_paths, mask_paths

def _get_test_dataset_items(data_folder):
    print("start getting test items...")
    img_paths = []
    for root, dirs, _ in os.walk(data_folder):
        for dir in dirs:
            foldername = os.path.basename(dir)
            imgpath = os.path.join(data_folder, dir, foldername+'_PAN.tif')
            if os.path.isfile(imgpath):
                img_paths.append(imgpath)
                print("pair "+foldername+" is done.")
            else:
                print('cannot find the image:', imgpath)
    logging.info('Found {} images in the folder {}'.format(len(img_paths), data_folder))
    return img_paths

if __name__ == '__main__':
    dataset = CustomSegmentation()
