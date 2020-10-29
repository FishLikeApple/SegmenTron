from __future__ import print_function

import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import logging
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import time
from PIL import Image

from tabulate import tabulate
from torchvision import transforms
from segmentron.utils.visualize import get_color_pallete
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.score import SegmentationMetric
from segmentron.utils.distributed import synchronize, make_data_sampler, make_batch_data_sampler
from segmentron.config import cfg
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup


class tester(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])

        # dataset and dataloader
        val_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='trainval', mode='test', transform=input_transform)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=cfg.TEST.BATCH_SIZE, drop_last=False)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)
        self.classes = val_dataset.classes
        # create network
        self.model = get_segmentation_model(self.device)

        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'named_modules') and \
            cfg.MODEL.BN_EPS_FOR_ENCODER:
            logging.info('set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
            self.set_batch_norm_attr(self.model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        self.model.to(self.device)

        self.metric = SegmentationMetric(val_dataset.num_class, args.distributed)

    def set_batch_norm_attr(self, named_modules, attr, value):
        for m in named_modules:
            if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                setattr(m[1], attr, value)

    def test(self):
        if not os.path.exists('output'):
            os.makedirs('output')
        
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model

        logging.info("Start test, Total sample: {:d}".format(len(self.val_loader)))

        time_start = time.time()
        for i, (image, filename) in enumerate(self.val_loader):
            image = image.to(self.device)

            with torch.no_grad():
                output = model.evaluate(image)

            for i in range(len(filename)):
                pred = torch.argmax(output[i], 0).squeeze(0).cpu().data.numpy()
                mask = Image.fromarray((pred*255).astype('uint8'))
                outname = filename[i] + '.png'
                mask.save(os.path.join('output', outname))


if __name__ == '__main__':
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()

    default_setup(args)

    evaluator = tester(args)
    evaluator.test()
