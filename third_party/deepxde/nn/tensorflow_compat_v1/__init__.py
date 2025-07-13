"""Package for tensorflow.compat.v1 NN modules."""

__all__ = [
    "DeepONet",
    "DeepONetCartesianProd",
    "Resolvent_DeepONet",
    "FNN",
    "MfNN",
    "MIONet",
    "MIONetCartesianProd",
    "MsFFN",
    "NN",
    "PFNN",
    "ResNet",
    "STMsFFN",
    "Transformer_decoder_deeponet",
    "MIONet_CNN_no_average",
    "decoder_deeponet_Transformer",
    "MIONet_CNN_no_average_resolvent",
    "DeepONet_resolvent",
    "D_DeepONet_resolvent",
    "DeepONet_resolvent_3d"
]

from .deeponet import DeepONet, DeepONetCartesianProd,Resolvent_DeepONet
from .fnn import FNN, PFNN
from .mfnn import MfNN
from .mionet import MIONet, MIONetCartesianProd,Transformer_decoder_deeponet,MIONet_CNN_no_average,decoder_deeponet_Transformer,MIONet_CNN_no_average_resolvent,DeepONet_resolvent,D_DeepONet_resolvent,DeepONet_resolvent_3d
from .msffn import MsFFN, STMsFFN
from .nn import NN
from .resnet import ResNet
