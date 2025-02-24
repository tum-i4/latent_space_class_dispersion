import torch
import torchvision.transforms.functional as FT

# from ssd.software.datasets.augmentation import SSDAugmentation


def preprocess(image, classes, coordinates, config, norm_mean, norm_std, augmentation=False):
    """
    Converts the given image and the bounding box coordinates and classes
    information to preprocessed PyTorch Tensors. The function needs to know
    whether this image should be augmented based on the configuration and
    the training or testing mode.

    SOURCE: https://github.com/pytorch/examples/blob/master/imagenet/main.py

    :param image: The image that should be preprocessed.
    :param classes: The category information for each of the bounding boxes that
            belong to this image.
    :param coordinates: The bounding coordinates in 'minmax' format.
    :param config: The configuration file for the training or testing script.
    :param augmentation: Whether the augmentation should be applied to the data.

    :return: The preprocessed image, its bounding box categories and the
            bounding box coordinates as PyTorch tensors.
    """
    width = image.width
    height = image.height

    classes = torch.tensor(classes, dtype=torch.long)
    coordinates = torch.tensor(coordinates, dtype=torch.float)
    coordinates = to_relative_coordinates(width, height, coordinates)

    image = FT.resize(image, (config.img_res, config.img_res))
    image = FT.to_tensor(image)

    # Checks if the image is greyscale and adds additional channels if so.
    if image.shape[0] == 1:
        image = image.view(1, config.img_res, config.img_res).expand(3, -1, -1)

    image = FT.normalize(image, mean=norm_mean, std=norm_std)

    return image, classes, coordinates


def to_relative_coordinates(width, height, coordinates):
    """
    Converts the given bounding box coordinates to relative coordinates between
    0.0 and 1.0. We basically normalize the input coordinates so that our
    network only needs to predict small values in the same range for all images.

    :param width: The width of the original image.
    :param height: The height of the original image.
    :param coordinates: A tensor of all bounding box coordinates for the image.

    :return: A tensor of coordinates that have been divided my the width and
            height of the image itself.
    """
    coordinates[:, [0, 2]] /= width
    coordinates[:, [1, 3]] /= height

    return coordinates
