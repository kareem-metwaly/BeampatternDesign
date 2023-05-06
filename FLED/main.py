"""This file is used for running / testing some building blocks."""
# from utils.config_classes import DatasetConfig
#
# output = DatasetConfig.parse_yaml("configs/dataset/dataset1.yaml")
#
# print(output)

from losses import LossRegistry
from models import ModelRegistry

our_loss = LossRegistry.get_class("NewLoss")
our_model = ModelRegistry.get_class("Model")
print(LossRegistry._registry)
print(ModelRegistry._registry)
print(our_loss)
