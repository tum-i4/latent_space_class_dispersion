import torch
from pathlib import Path

op_dir = "gtsrb_new_merged_1"

fuzzing_path = Path(
    "./models/gtsrb/gtsrb_new/weights/pytorch_classification_gtsrb_new.pth"
)

cur_exp_path = Path(
    "./results/", op_dir + "/trained_mutants/AAA_Original_000/model_script.pth",
    map_location="cpu",
)

op_path = Path(
    # "./models/mnist/lenet5/weights/pytorch_classification_mnist_lenet5_updated.pth",
    "./results/", op_dir + "/trained_mutants/AAA_Original_000/model_fuzzing_new.pth",
)

cur_model_script = torch.jit.load(cur_exp_path, map_location="cpu")

old_weights = torch.load(fuzzing_path) # Old_keys to be replace and weights to be used.
cur_weights_keys = list(cur_model_script.state_dict().keys()) # Current_keys

new_dict = dict(zip(cur_weights_keys, old_weights.values()))
print(old_weights.keys(), new_dict.keys())

# for i, key in enumerate(old_weights.keys()):
#     new_state_dict[new_key_list[i]] = old_weights[key]

print("New weights stored at:", op_path)

torch.save(new_dict, op_path)
