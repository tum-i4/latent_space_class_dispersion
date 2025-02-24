from collections import OrderedDict

import torch.nn as nn
from torch_intermediate_layer_getter import IntermediateLayerGetter


class ClassificationModel(nn.Module):
    def intermediate_outputs(self, data):
        """
        generates outputs for the selected intermediate layers and save the layers names and number of neurons as class
        variable.
        NOTE: This function should be called in each run at the beginning for the fault detection to work.
        :param data: any input data
        :return: intermediate layer outputs
        """
        intermediate_layers = OrderedDict()
        self.num_neurons = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Sequential) or isinstance(module, self.__class__):
                continue
            p = list(module.parameters())
            if not len(p) == 0:
                self.num_neurons.append(list(module.parameters())[-1].size(0))
                intermediate_layers.update({name: name})
        self.intermediate_layers = list(intermediate_layers.keys())
        assert len(self.num_neurons) == len(self.intermediate_layers), "Intermediate layer identification going wrong"
        intermediate_getter = IntermediateLayerGetter(model=self, return_layers=intermediate_layers, keep_output=True)
        mids, _ = intermediate_getter(data)
        return mids
