import copy
import inspect
import random
import types
from typing import Dict
import ast

import torch
import torch.nn as nn
from mutagen.mut_base import PostTrainingMutation

class _LayerAdderTransformer(ast.NodeTransformer):
    def __init__(self, target_layer_name, new_layer_name, new_layer_def):
        self.target_layer_name = target_layer_name
        self.new_layer_name = new_layer_name
        self.new_layer_def = new_layer_def
        self.layer_added = False  # Track if layer is already added

    def visit_FunctionDef(self, node):
        # Modify __init__ method to add new layer
        if node.name == "__init__":
            new_layer_assign = ast.Assign(
                targets=[ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr=self.new_layer_name, ctx=ast.Store())],
                value=ast.parse(self.new_layer_def).body[0].value  # Parse the new layer definition
            )
            self.set_node_metadata(new_layer_assign, node.body[-1])  # Set metadata from last node
            node.body.append(new_layer_assign)  # Add the new layer to __init__
            self.layer_added = True

        # Modify forward function
        if node.name == "forward":
            new_body = []
            for stmt in node.body:
                new_body.append(stmt)

                if isinstance(stmt, ast.Assign) and self.contains_target_layer(stmt.value, self.target_layer_name):
                    new_assign = ast.Assign(
                        targets=[ast.Name(id='x', ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr=self.new_layer_name, ctx=ast.Load()),
                            args=[ast.Name(id='x', ctx=ast.Load())],
                            keywords=[]
                        )
                    )
                    self.set_node_metadata(new_assign, stmt)  # Set metadata from the current statement
                    new_body.append(new_assign)

            node.body = new_body  # Update the function body
        return node

    def contains_target_layer(self, node, target_layer_name):
        """ Recursively search for the target layer name in nested function calls """
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "self" and node.func.attr == target_layer_name:
                    return True

            # Recursively check within function arguments
            for arg in node.args:
                if self.contains_target_layer(arg, target_layer_name):
                    return True

        return False

    def set_node_metadata(self, new_node, ref_node):
        """ Copies line number and column offset metadata to new AST node """
        new_node.lineno = ref_node.lineno
        new_node.col_offset = ref_node.col_offset
        new_node.end_lineno = ref_node.end_lineno
        new_node.end_col_offset = ref_node.end_col_offset


class GaussianFuzzing(PostTrainingMutation):
    def __init__(self, scale: float = 0.01):
        super().__init__()
        self.scale = float(scale)

    def mutate(self, model: nn.Module, ast_module: ast.Module) -> nn.Module:
        for name, param in model.named_parameters():
            if param.requires_grad:
                noise = torch.randn(param.size()) * self.scale
                param.data += noise
        
        return model, ast_module
    

class WeightShuffling(PostTrainingMutation):
    """
    A mutation operation that shuffles the weights of a specified layer or neuron in a neural network model.
    1. If only layer_index is specified, the weights of all neurons on the specified layer will be shuffled.
    2. If only neuron_index is specified, the weights of the specified neuron in each layer will be shuffled.
    3. If both layer_index and neuron_index are specified, the weights of the specified neuron in the specified layer will be shuffled.
    4. If nothing is specified, the weights of all neurons in all layers will be shuffled.
    5. If random_neuron is set to True, a random neuron will be selected for mutation.
    Attributes:
        layer_index (int, optional): The index of the layer to mutate. If None, all layers will be considered.
        neuron_index (int, optional): The index of the neuron to mutate. If None, all neurons will be considered.
        random_neuron (bool): If True, a random neuron will be selected for mutation.

    Note: Only Conv2d and Linear layers are considered for layer_index.
    For example, Conv2d->Relu->BatchNorm2d->Linear will be considered as two layers and layer_index=1 will refer to the Linear layer.
    """
    def __init__(self, layer_index: int = None, neuron_index: int = None, random_neuron: bool = False):
        super().__init__()
        self.layer_index = int(layer_index) if layer_index is not None else None
        self.neuron_index = int(neuron_index) if neuron_index is not None else None
        self.random_neuron = bool(random_neuron)

    def mutate(self, model: nn.Module, ast_module: ast.Module) -> nn.Module:
        layers = [layer for layer in model.children() if isinstance(layer, (nn.Conv2d, nn.Linear))]
        if self.random_neuron:
            self.layer_index = random.randint(0, len(layers) - 1)
            self.neuron_index = random.randint(0, layers[self.layer_index].weight.size(0) - 1)
            print("# A random neuron was selected for mutation.")

        print(f"# layer_index: {self.layer_index}, neuron_index: {self.neuron_index}")
        
        if self.layer_index is not None:
            if not (0 <= self.layer_index < len(layers)):
                raise IndexError("layer_index out of bounds")
            param = layers[self.layer_index].weight
            if param.requires_grad:
                if self.neuron_index is not None:
                    if not (0 <= self.neuron_index < param.size(0)):
                        raise IndexError("neuron_index out of bounds")
                    param_data = param.data[self.neuron_index].view(-1)
                    indices = torch.randperm(param_data.size(0))
                    param.data[self.neuron_index] = param_data[indices].view(param[self.neuron_index].size())
                    print("# The weights of the specified neuron in the specified layer were shuffled.")
                else:
                    param_data = param.data.view(-1)
                    indices = torch.randperm(param_data.size(0))
                    param.data = param_data[indices].view(param.size())
                    print("# The weights of all neurons on the specified layer were shuffled.")
        else:
            for layer in layers:
                param = layer.weight
                if param.requires_grad:
                    if self.neuron_index is not None:
                        if not (0 <= self.neuron_index < param.size(0)):
                            raise IndexError("neuron_index out of bounds")
                        param_data = param.data[self.neuron_index].view(-1)
                        indices = torch.randperm(param_data.size(0))
                        param.data[self.neuron_index] = param_data[indices].view(param[self.neuron_index].size())
                        print("# The weights of the specified neuron in each layer were shuffled.")
                    else:
                        param_data = param.data.view(-1)
                        indices = torch.randperm(param_data.size(0))
                        param.data = param_data[indices].view(param.size())
                        print("# The weights of all neurons in all layers were shuffled.")
        return model, ast_module


class NeuronEffectBlocking(PostTrainingMutation):
    """
    A mutation operation that blocks the effect of specific neurons in a neural network model by zeroing out their weights.
    1. If only layer_index is specified, the weights of all neurons on the specified layer will be zeroed out.
    2. If only neuron_index is specified, the weights of the specified neuron in each layer will be zeroed out.
    3. If both layer_index and neuron_index are specified, the weights of the specified neuron in the specified layer will be zeroed out.
    4. If nothing is specified, the weights of all neurons in all layers will be zeroed out.
    5. If random_neuron is set to True, a random neuron will be selected for mutation.
    Attributes:
        layer_index (int, optional): The index of the layer to mutate. If None, all layers are considered.
        neuron_index (int, optional): The index of the neuron to mutate. If None, all neurons in the specified layer(s) are considered.
        random_neuron (bool): If True, a random neuron in a random layer is selected for mutation.
    
    Note: Only Conv2d and Linear layers are considered for layer_index.
    For example, Conv2d->Relu->BatchNorm2d->Linear will be considered as two layers and layer_index=1 will refer to the Linear layer.
    """
    def __init__(self, layer_index: int = None, neuron_index: int = None, random_neuron: bool = False):
        super().__init__()
        self.layer_index = int(layer_index) if layer_index is not None else None
        self.neuron_index = int(neuron_index) if neuron_index is not None else None
        self.random_neuron = bool(random_neuron)

    def mutate(self, model: nn.Module, ast_module: ast.Module) -> nn.Module:
        layers = [layer for layer in model.children() if isinstance(layer, (nn.Conv2d, nn.Linear))]
        if self.random_neuron:
            self.layer_index = random.randint(0, len(layers) - 1)
            self.neuron_index = random.randint(0, layers[self.layer_index].weight.size(0) - 1)
            print("# A random neuron was selected for mutation.")
        
        print(f"# layer_index: {self.layer_index}, neuron_index: {self.neuron_index}")

        if self.layer_index is not None:
            if not (0 <= self.layer_index < len(layers)):
                raise IndexError("layer_index out of bounds")
            param = layers[self.layer_index].weight
            if param.requires_grad:
                if self.neuron_index is not None:
                    if not (0 <= self.neuron_index < param.size(0)):
                        raise IndexError("neuron_index out of bounds")
                    param.data[self.neuron_index].zero_()
                    print("# The weights of the specified neuron in the specified layer were zeroed out.")
                else:
                    param.data.zero_()
                    print("# The weights of all neurons on the specified layer were zeroed out.")
        else:
            for layer in layers:
                param = layer.weight
                if param.requires_grad:
                    if self.neuron_index is not None:
                        if not (0 <= self.neuron_index < param.size(0)):
                            raise IndexError("neuron_index out of bounds")
                        param.data[self.neuron_index].zero_()
                        print("# The weights of the specified neuron in each layer were zeroed out.")
                    else:
                        param.data.zero_()
                        print("# The weights of all neurons in all layers were zeroed out.")
        return model, ast_module


class NeuronActivationInverse(PostTrainingMutation):
    """
    A mutation operation that inverts the activation of neurons in a specified layer of a neural network model.
    1. If only layer_index is specified, the activation of all neurons on the specified layer will be inverted.
    2. If only neuron_index is specified, the activation of the specified neuron in each layer will be inverted.
    3. If both layer_index and neuron_index are specified, the activation of the specified neuron in the specified layer will be inverted.
    4. If nothing is specified, the activation of all neurons in all layers will be inverted.
    5. If random_neuron is set to True, a random neuron will be selected for mutation.
    Attributes:
        layer_index (int, optional): The index of the layer to mutate. If None, all layers are considered.
        neuron_index (int, optional): The index of the neuron to mutate. If None, all neurons in the specified layer are considered.
        random_neuron (bool): If True, a random neuron is selected for mutation.

    Note: Only Conv2d and Linear layers are considered for layer_index.
    For example, Conv2d->Relu->BatchNorm2d->Linear will be considered as two layers and layer_index=1 will refer to the Linear layer.
    """
    def __init__(self, layer_index: int = None, neuron_index: int = None, random_neuron: bool = False):
        super().__init__()
        self.layer_index = int(layer_index) if layer_index is not None else None
        self.neuron_index = int(neuron_index) if neuron_index is not None else None
        self.random_neuron = bool(random_neuron)

    def mutate(self, model: nn.Module, ast_module: ast.Module) -> nn.Module:
        layers = [layer for layer in model.children() if isinstance(layer, (nn.Conv2d, nn.Linear))]
        if self.random_neuron:
            self.layer_index = random.randint(0, len(layers) - 1)
            self.neuron_index = random.randint(0, layers[self.layer_index].weight.size(0) - 1)
            print("# A random neuron was selected for mutation.")
        
        print(f"# layer_index: {self.layer_index}, neuron_index: {self.neuron_index}")

        if self.layer_index is not None:
            if not (0 <= self.layer_index < len(layers)):
                raise IndexError("layer_index out of bounds")
            param = layers[self.layer_index].weight
            if param.requires_grad:
                if self.neuron_index is not None:
                    if not (0 <= self.neuron_index < param.size(0)):
                        raise IndexError("neuron_index out of bounds")
                    param.data[self.neuron_index] = -param.data[self.neuron_index]
                    print("# The activation of the specified neuron in the specified layer was inverted.")
                else:
                    param.data = -param.data
                    print("# The activation of all neurons on the specified layer was inverted.")
        else:
            for layer in layers:
                param = layer.weight
                if param.requires_grad:
                    if self.neuron_index is not None:
                        if not (0 <= self.neuron_index < param.size(0)):
                            raise IndexError("neuron_index out of bounds")
                        param.data[self.neuron_index] = -param.data[self.neuron_index]
                        print("# The activation of the specified neuron in each layer was inverted.")
                    else:
                        param.data = -param.data
                        print("# The activation of all neurons in all layers was inverted.")
        return model, ast_module


class NeuronSwitch(PostTrainingMutation):
    """
    A mutation operation that switches the weights of two randomly selected neurons in a specified layer
    or in all layers of a neural network model.
    1. If only layer_index is specified, the weights of two randomly selected neurons on the specified layer will be switched.
    2. If nothing is specified, the weights of two randomly selected neurons in all layers will be switched.
    3. If random_layer is set to True, the weights of two randomly selected neurons on a random layer will be switched.
    Attributes:
        layer_index (int, optional): The index of the layer to mutate. If None, all layers will be mutated.
        random_layer (bool): If True, a random layer will be selected for mutation.

    Note: Only Conv2d and Linear layers are considered for layer_index.
    For example, Conv2d->Relu->BatchNorm2d->Linear will be considered as two layers and layer_index=1 will refer to the Linear layer.
    """
    def __init__(self, layer_index: int = None, random_layer: bool = False):
        super().__init__()
        self.layer_index = int(layer_index) if layer_index is not None else None
        self.random_layer = random_layer

    def mutate(self, model: nn.Module, ast_module: ast.Module) -> nn.Module:
        layers = [layer for layer in model.children() if isinstance(layer, (nn.Conv2d, nn.Linear))]
        if self.random_layer:
            self.layer_index = random.randint(0, len(layers) - 1)
            print("# A random layer was selected for mutation.")
        
        print(f"# layer_index: {self.layer_index}")

        if self.layer_index is not None:
            if not (0 <= self.layer_index < len(layers)):
                raise IndexError("layer_index out of bounds")
            param = layers[self.layer_index].weight
            if param.requires_grad:
                idx1, idx2 = random.sample(range(param.size(0)), 2)
                param.data[[idx1, idx2], :] = param.data[[idx2, idx1], :]
                print("# The weights of two randomly selected neurons on the specified layer were switched.")
                print(f"# Switched neurons: {idx1} and {idx2}")
        else:
            for layer in layers:
                param = layer.weight
                if param.requires_grad:
                    idx1, idx2 = random.sample(range(param.size(0)), 2)
                    param.data[[idx1, idx2], :] = param.data[[idx2, idx1], :]
                    print("# The weights of two randomly selected neurons in each layer were switched.")
                    print(f"# Switched neurons: {idx1} and {idx2}")
        return model, ast_module


class LayerAddition(PostTrainingMutation):
    """
    A mutation operation that adds a new layer to a neural network model after training.
    1. If layer_index is specified, the new layer is inserted at the specified index.
    2. If layer_index is None or random_position is set to True, the new layer is inserted at a random index.
    Attributes:
        layer_index (int, optional): The index at which to insert the new layer. If None, a random index is chosen.

    # Creates a new layer with the same number of input and output features as the layer one after the layer at the specified index
    # If the layer is a Conv2d layer, insert a new Conv2d layer after it with the same number of input and output features
    # otherwise, insert a new Linear layer after it with the same number of input and output features
    # NOTE: The idea of adding Conv2d layer after Conv2d is based on easiness of implementation. It can be changed to Linear layer after Conv2d. With this implementation we dont need to calculate the linear layer input size based on conv2d's 2-dimensional output shape.

    Note: Only Conv2d and Linear layers are considered for layer_index.
    For example, Conv2d->Relu->BatchNorm2d->Linear will be considered as two layers and layer_index=1 will refer to the Linear layer.
    """
    def __init__(self, layer_index: int = None, random_position: bool = False):
        super().__init__()
        self.layer_index = int(layer_index) if layer_index is not None else None
        self.random_position = bool(random_position)

    def mutate(self, model: nn.Module, ast_module: ast.Module) -> nn.Module:
        all_layers = list(model.named_children())
        conv_linear_layers = []
        conv_linear_layer_names = []
        for name, layer in all_layers:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                conv_linear_layers.append(layer)
                conv_linear_layer_names.append(name)
        
        if len(conv_linear_layers) > 0:
            if self.layer_index is not None and self.random_position is False:
                if not (0 <= self.layer_index < len(conv_linear_layers)):
                    raise IndexError("layer_index out of bounds")
                insert_position = self.layer_index
            else:
                insert_position = random.randint(0, len(conv_linear_layers) - 1)
                print("# A random position was selected for layer insertion.")
            
            print(f"# insert_position: {insert_position}")

            target_layer = conv_linear_layers[insert_position]
            target_layer_name = conv_linear_layer_names[insert_position]
            if isinstance(target_layer, nn.Conv2d):
                features = target_layer.out_channels
                # kernel_size is decided arbitrarily and stride=1, padding=1 are used to maintain the dimensionality of the input such that the model does not break
                new_layer = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
                new_layer_name = "new_conv2d_layer"
                new_layer_def = "nn.Conv2d({}, {}, kernel_size=3, stride=1, padding=1)".format(features, features)
            elif isinstance(target_layer, nn.Linear):
                features = target_layer.out_features
                new_layer = nn.Linear(features, features)
                new_layer_name = "new_linear_layer"
                new_layer_def = "nn.Linear({}, {})".format(features, features)
            else:
                raise ValueError("Unsupported layer type for insertion")
            
            # Insert the new layer into the model

            # Add the new layer as an attribute
            setattr(model, new_layer_name, new_layer)

            # Modify the AST module to add the new layer and update the forward function
            transformer = _LayerAdderTransformer(target_layer_name, new_layer_name, new_layer_def)
            ast_module = transformer.visit(ast_module)

            return model, ast_module


# NOTE - Following operations will not always work if there is no suitable choice that won't break the model. Therefore, they are not included in the final implementation.

# class LayerRemoval(PostTrainingMutation):
#     def mutate(self, model: nn.Module) -> nn.Module:
#         layers = list(model.children())
#         if len(layers) > 1:
#             layer_to_remove = random.choice(layers)
#             model._modules = {k: v for k, v in model._modules.items() if v != layer_to_remove}
#         return model


# class LayerDuplication(PostTrainingMutation):
#     def mutate(self, model: nn.Module) -> nn.Module:
#         layers = list(model.children())
#         if len(layers) > 0:
#             layer_to_duplicate = random.choice(layers)
#             new_layer = nn.Linear(layer_to_duplicate.in_features, layer_to_duplicate.out_features)
#             model.add_module(f"{layer_to_duplicate}_dup", new_layer)
#         return model
