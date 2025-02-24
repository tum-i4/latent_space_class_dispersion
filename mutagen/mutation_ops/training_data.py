import torch
from typing import Dict


def remove_data(data: Dict[str, torch.Tensor], to_remove: int):
    data['img_train'] = data['img_train'][:to_remove]
    data['lbl_train'] = data['lbl_train'][:to_remove]


def add_noise(data: Dict[str, torch.Tensor], percentage: float, mean: float = 0.0, std_percentage: float = 0.3):
    tensor = data['img_train']

    images_to_change = round(percentage * tensor.size(0))
    noise_indices = torch.randperm(tensor.size(0))[:images_to_change]

    std_deviation = torch.std(tensor.flatten()) * std_percentage
    noise = torch.randn_like(tensor) * std_deviation + mean

    tensor[noise_indices] += noise[noise_indices]
    torch.clamp_(tensor, 0, 1)


def make_classes_overlap(data: Dict[str, torch.Tensor], percentage: float):
    images = data['img_train']
    labels = data['lbl_train']

    counts = torch.unique(labels, return_counts=True)
    d = sorted([(count, lbl) for lbl, count in zip(counts[0], counts[1])])

    mc_indices = (labels == d[-1][1]).nonzero(as_tuple=False)
    n = round(len(mc_indices) * percentage)

    selected_indices = mc_indices[torch.randperm(len(mc_indices))[:n]]

    new_labels = labels.new_full((n,), d[-2][1])

    data['img_train'] = torch.cat([images[selected_indices].squeeze(dim=1), images], dim=0)
    data['lbl_train'] = torch.cat([new_labels, labels], dim=0)


def remove_samples(data: Dict[str, torch.Tensor], percentage: float):
    images = data['img_train']
    labels = data['lbl_train']
    num_classes = data['num_classes']

    mask = torch.ones(labels.numel(), dtype=torch.bool, device=images.device)

    for label in range(num_classes):
        indices = (labels == label).nonzero(as_tuple=False)

        samples_to_remove = int(len(indices) * percentage)

        if samples_to_remove > 0:
            selected_indices = torch.randperm(len(indices), device=images.device)[:samples_to_remove]
            mask[indices[selected_indices]] = False

    data['img_train'] = images[mask]
    data['lbl_train'] = labels[mask]


def change_labels(data: Dict[str, torch.Tensor], percentage: float, which_label: int = 0):
    tensor = data['lbl_train']
    num_classes = data['num_classes']

    change_indices = torch.nonzero(tensor == which_label, as_tuple=False)

    num_values_to_change = int(len(change_indices) * percentage)
    indices_to_change = torch.randperm(len(change_indices))[:num_values_to_change]

    new_values = torch.randint(0, num_classes, size=(num_values_to_change,), dtype=torch.uint8, device=tensor.device)

    tensor[change_indices[indices_to_change].flatten()] = new_values
