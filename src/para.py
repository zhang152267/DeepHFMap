import model.deephfmap as deephfmap
import torch
import torch.nn as nn
net = deephfmap.DeepHFMap()
batch=4
epo = 1000
l1=0.00
mod = "DeepHFMap/ya".format(epo,batch)
