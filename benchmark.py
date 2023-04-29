#!/usr/bin/python
# -*- coding: utf-8 -*-
from time import time
import numpy as np
import torch.cuda
from torch import nn

def get_pdn(out=384):
    return nn.Sequential(
        nn.Conv2d(3, 256, 4), nn.ReLU(inplace=True),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(256, 512, 4), nn.ReLU(inplace=True),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(512, 512, 1), nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, 3), nn.ReLU(inplace=True),
        nn.Conv2d(512, out, 4), nn.ReLU(inplace=True),
        nn.Conv2d(out, out, 1)
    )

def get_ae():
    return nn.Sequential(
        # encoder
        nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 8),
        # decoder
        nn.Upsample(3, mode='bilinear'),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True),
        nn.Upsample(8, mode='bilinear'),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True),
        nn.Upsample(15, mode='bilinear'),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True),
        nn.Upsample(32, mode='bilinear'),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True),
        nn.Upsample(63, mode='bilinear'),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True),
        nn.Upsample(127, mode='bilinear'),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True),
        nn.Upsample(56, mode='bilinear'),
        nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True),
        nn.Conv2d(64, 384, 3, 1, 1)
    )


gpu = torch.cuda.is_available()

autoencoder = get_ae()
teacher = get_pdn(384)
student = get_pdn(768)

autoencoder = autoencoder.eval()
teacher = teacher.eval()
student = student.eval()

if gpu:
    autoencoder.half().cuda()
    teacher.half().cuda()
    student.half().cuda()

quant_mult = torch.e
quant_add = torch.pi
with torch.no_grad():
    times = []
    for rep in range(2000):
        image = torch.randn(1, 3, 256, 256, dtype=torch.float16 if gpu else torch.float32)
        start = time()
        if gpu:
            image = image.cuda()

        t = teacher(image)
        s = student(image)

        st_map = torch.mean((t - s[:, :384]) ** 2, dim=1)
        ae = autoencoder(image)
        ae_map = torch.mean((ae - s[:, 384:]) ** 2, dim=1)
        st_map = st_map * quant_mult + quant_add
        ae_map = ae_map * quant_mult + quant_add
        result_map = st_map + ae_map
        result_on_cpu = result_map.cpu().numpy()
        timed = time() - start
        times.append(timed)
print(np.mean(times[-1000:]))

