from domino import explore, DominoSlicer
import meerkat as mk

import os

dp =  mk.datasets.get("imagenette", dataset_dir="C:/Users/rohil/Downloads/Uni Bonn/WiSe 2022-23/BigDataLab_Domino/BigDataLab_Domino/src/meerkat-main/meerkat/datasets/imagenet/")

# we'll only be using the validation data
dp = dp.lz[dp["split"] == "valid"]

print(dp.shape)