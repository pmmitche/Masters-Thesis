import timm
import torch
import numpy as np
from torchsummary import summary

from nni.compression.pytorch.pruning import L1NormPruner
from nni.compression.pytorch.speedup import ModelSpeedup


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_exclude():
    batch_size = 16
    inference_input = torch.randn(batch_size, 3, 360, 640).to(device)

    sparsity = 0.8
    model = timm.create_model('efficientnet_lite0', pretrained=True)
    model.to(device)
    print("Model Structure...")
    print(model)

    print("\nStarting Pruning Process...")
    config_list = None
    # create pruned model
    config_list = [{
        'sparsity_per_layer': sparsity,
        'op_types': ['Linear', 'Conv2d']
    }, {
        'exclude': True,
        'op_names': ['conv_stem']
    }]

    print("\nConfig List:", config_list)

    dummy_input = torch.rand(1, 3, 360, 640).to(device)
    pruner = L1NormPruner(model, config_list)

    # compress the model and generate the masks
    _, masks = pruner.compress()

    # need to unwrap the model, if the model is wrapped before speedup
    pruner._unwrap_model()

    # speedup the model, for more information about speedup, please refer :doc:`pruning_speedup`.
    ModelSpeedup(model, dummy_input, masks).speedup_model()

    print("\n\n----------- Model Summary: Pruned at {}% with NNI -----------\n".format(sparsity * 100))
    if torch.cuda.is_available():
        model.cuda()
    summary(model, (3, 360, 640))


test_exclude()
