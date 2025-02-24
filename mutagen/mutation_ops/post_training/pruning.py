import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from mutagen.mut_base import PostTrainingMutation

############################################################
#Local Pruning
############################################################


#prunes 30% of the weights in all layers based on the L1 norm

# Local Pruning: Unstructured weight pruning per layer
class LocalWeightPruningOperator(PostTrainingMutation):
    def __init__(self, prune_ratio: float = 0.8):
        super().__init__()
        self.prune_ratio = float(prune_ratio)

    def mutate(self, model: nn.Module) -> nn.Module:
        # Convert TorchScript model back to regular PyTorch model
        if isinstance(model, torch.jit.ScriptModule):
            # Create a copy of the model architecture
            from models.mnist.lenet5.model import Net  # Import your model class
            pytorch_model = Net()
            # Copy the state dict
            pytorch_model.load_state_dict(model.state_dict())
            model = pytorch_model

        pruned = False
        for name, module in model.named_modules():
            if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
                module.weight = torch.nn.Parameter(module.weight)
                prune.l1_unstructured(module, name='weight', amount=self.prune_ratio)
                prune.remove(module, 'weight')
                pruned = True

        if not pruned:
            print("LocalWeightPruningOperator: No layers found to prune. Skipping pruning.")
       

        # Convert back to TorchScript
        return torch.jit.script(model)



# Prunes weights and biases while preserving first and last n layers
class WeightandBiasPruningOperator(PostTrainingMutation):
    def __init__(self, prune_ratio_weights: float = 0.3, prune_ratio_biases: float = 0.1):
        super().__init__()
        self.prune_ratio_weights = float(prune_ratio_weights)
        self.prune_ratio_biases = float(prune_ratio_biases)

    def mutate(self, model: nn.Module) -> nn.Module:
        # Convert TorchScript model back to regular PyTorch model
        if isinstance(model, torch.jit.ScriptModule):
            from models.mnist.lenet5.model import Net
            pytorch_model = Net()
            pytorch_model.load_state_dict(model.state_dict())
            model = pytorch_model

        pruned = False
        for name, module in model.named_modules():
            if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
                module.weight = torch.nn.Parameter(module.weight)
                prune.l1_unstructured(module, name='weight', amount=self.prune_ratio_weights)
                prune.remove(module, 'weight')
                
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias = torch.nn.Parameter(module.bias)
                    prune.l1_unstructured(module, name='bias', amount=self.prune_ratio_biases)
                    prune.remove(module, 'bias')
                
                pruned = True
              

        if not pruned:
            print("WeightandBiasPruningOperator: No layers found to prune. Skipping pruning.")
       

        # Convert back to TorchScript
        return torch.jit.script(model)


# # Prunes convolution and linear layers differently
class PruningTwoParamsOperator(PostTrainingMutation):
    def __init__(self, prune_ratio_conv: float = 0.2, prune_ratio_linear: float = 0.4):
        super().__init__()
        self.prune_ratio_conv = float(prune_ratio_conv)
        self.prune_ratio_linear = float(prune_ratio_linear)

    def mutate(self, model: nn.Module) -> nn.Module:
        # Convert TorchScript model back to regular PyTorch model
        if isinstance(model, torch.jit.ScriptModule):
            from models.mnist.lenet5.model import Net
            pytorch_model = Net()
            pytorch_model.load_state_dict(model.state_dict())
            model = pytorch_model

        pruned = False
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                module.weight = torch.nn.Parameter(module.weight)
                prune.l1_unstructured(module, name='weight', amount=self.prune_ratio_conv)
                prune.remove(module, 'weight')
                pruned = True
            elif isinstance(module, nn.Linear):
                module.weight = torch.nn.Parameter(module.weight)
                prune.l1_unstructured(module, name='weight', amount=self.prune_ratio_linear)
                prune.remove(module, 'weight')
                pruned = True

        if not pruned:
            print("PruningTwoParamsOperator: No convolution or linear layers found. Skipping pruning.")
       

        # Convert back to TorchScript
        return torch.jit.script(model)


# # Structured pruning (L2 norm) for input and output channels
class StructuredPruningOperator(PostTrainingMutation):
     def __init__(self, prune_ratio: float = 0.3):
         super().__init__()
         self.prune_ratio = float(prune_ratio)

     def mutate(self, model: nn.Module) -> nn.Module:
        if isinstance(model, torch.jit.ScriptModule):
            from models.mnist.lenet5.model import Net
            pytorch_model = Net()
            pytorch_model.load_state_dict(model.state_dict())
            model = pytorch_model
        pruned = False
        for name, module in model.named_modules():
            if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
                module.weight = torch.nn.Parameter(module.weight)
                prune.ln_structured(module, name='weight', amount=self.prune_ratio, n=2, dim=0)  # Output
                prune.ln_structured(module, name='weight', amount=self.prune_ratio, n=2, dim=1)
                prune.remove(module, 'weight')
                pruned = True

        if not pruned:
            print("StructuredPruningOperator: No eligible layers found. Skipping pruning.")
        return model


# # Local threshold pruning (absolute threshold)
class LocalThresholdPruningOperator(PostTrainingMutation):
    def __init__(self, threshold: float = 1):
        super().__init__()
        self.threshold = float(threshold)

    def mutate(self, model: nn.Module) -> nn.Module:
        if isinstance(model, torch.jit.ScriptModule):
            from models.mnist.lenet5.model import Net
            pytorch_model = Net()
            pytorch_model.load_state_dict(model.state_dict())
            model = pytorch_model
        pruned = False
        for name, module in model.named_modules():
            if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
                module.weight = torch.nn.Parameter(module.weight)
                mask = torch.abs(module.weight) > self.threshold
                prune.custom_from_mask(module, name='weight', mask=mask)
                prune.remove(module, 'weight')
                pruned = True

        if not pruned:
            print("LocalThresholdPruningOperator: No eligible weights found. Skipping pruning.")
        return model




# # Global threshold-based pruning
class GlobalThresholdPruningOperator(PostTrainingMutation):
    def __init__(self, threshold: float = 1):
        super().__init__()
        self.threshold = float(threshold)

    class ThresholdPruning(prune.BasePruningMethod):
        PRUNING_TYPE = "unstructured"

        def __init__(self, threshold):
            self.threshold = threshold

        def compute_mask(self, tensor, default_mask):
            return torch.abs(tensor) > self.threshold

    def mutate(self, model: nn.Module) -> nn.Module:
        if isinstance(model, torch.jit.ScriptModule):
            from models.mnist.lenet5.model import Net
            pytorch_model = Net()
            pytorch_model.load_state_dict(model.state_dict())
            model = pytorch_model

        print(f"Starting global threshold pruning with threshold: {self.threshold}")
        
        parameters_to_prune = []
        for name, module in model.named_modules():
            if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
                print(f"Found prunable layer: {name}")
                print(f"Before pruning - Layer: {name}, Weight sum: {module.weight.sum().item()}")
                parameters_to_prune.append((module, 'weight'))

        if not parameters_to_prune:
            print("GlobalThresholdPruningOperator: No layers found to prune. Skipping pruning.")
            return model

        print(f"Total layers to prune: {len(parameters_to_prune)}")

        for module, name in parameters_to_prune:
            module.weight = torch.nn.Parameter(module.weight)
            
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=GlobalThresholdPruningOperator.ThresholdPruning,
            threshold=self.threshold
        )
        
        for module, _ in parameters_to_prune:
            prune.remove(module, 'weight')
            print(f"After pruning - Layer: {module.__class__.__name__}, Weight sum: {module.weight.sum().item()}")

        print("Global threshold pruning completed successfully.")
        
        # Convert back to TorchScript
        return torch.jit.script(model)


# # Global L2 norm-based pruning
# class GlobalWeightPruningOperator(PostTrainingMutation):
#     def __init__(self, prune_ratio: float = 0.3):
#         super().__init__()
#         self.prune_ratio = prune_ratio

#     def mutate(self, model: nn.Module) -> nn.Module:
#         parameters_to_prune = [
#             (module, 'weight') for name, module in model.named_modules()
#             if isinstance(module, (nn.Linear, nn.Conv2d))
#         ]

#         if not parameters_to_prune:
#             print("GlobalWeightPruningOperator: No layers found to prune. Skipping pruning.")
#             return model

#         prune.global_unstructured(
#             parameters_to_prune,
#             pruning_method=prune.LnStructured,
#             amount=self.prune_ratio,
#             n=2, dim=1  # Input channels
#         )
#         prune.global_unstructured(
#             parameters_to_prune,
#             pruning_method=prune.LnStructured,
#             amount=self.prune_ratio,
#             n=2, dim=0  # Output channels
#         )
#         for module, name in parameters_to_prune:
#             prune.remove(module, name)
#         return model
