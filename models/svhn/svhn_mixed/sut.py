import os
import sys
import torch
from pathlib import Path
from typing import List, Optional, Tuple, Iterable, Union
import importlib

import numpy as np

from torchvision import transforms as tf


class SUT:
    def __init__(self, sut_folder: Optional[Path] = None, device: Optional[str] = None):
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'

        if sut_folder is None:
            sut_folder = Path(__file__).parent

        # self.net = torch.jit.load(sut_folder / "model_script.pth", map_location='cpu')
        state_dict = torch.load(sut_folder / "model.pth", map_location=self.device)

        spec = importlib.util.spec_from_file_location("model_svhn", sut_folder / "model.py")
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        self.net = model_module.Net()
        #self.net.load_state_dict(state_dict)
        
        sut_folder_name = str(sut_folder)
        if sut_folder_name.split("/")[-1] == "AAA_Original_000":
            temp_weights = torch.load(sut_folder / "model_fuzzing_new.pth") # To be compitable with Fuzzing baseline model.
            self.net.load_state_dict(temp_weights) # Convert weights according to new weights
        else:
            self.net.load_state_dict(state_dict)        
        self.net.eval()

        self.transform = tf.Compose([
            tf.ToPILImage(),
            tf.Resize((32, 32)),
            # tf.Grayscale(num_output_channels=1),
            tf.ToTensor(),
            # tf.Normalize(mean=(0.130,7), std=(0.3081,)) # Stored files are already normalized
        ])

    def __enter__(self):
        self.net.to(self.device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.net.to('cpu')

    def execute(self, images: Union[torch.Tensor, Iterable[np.array]]) -> List[Tuple[int, float, List[float]]]:
        with torch.no_grad():
            converted = [self.transform(img)[None, :] for img in images]
            images = torch.cat(converted)

            probabilities, classes, outputs = self.execute_raw(images)

            full_out = [
                (cls.item(), p.item(), out.detach().cpu().numpy().tolist())
                for cls, p, out in zip(classes, probabilities, outputs)
            ]

            return full_out

    def execute_raw(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            outputs = self.net(images.to(self.device))
            probabilities, classes = torch.log_softmax(outputs, dim=1).max(dim=1)
        return probabilities, classes, outputs


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    img = plt.imread(sys.argv[1])

    with SUT() as s:
        r = s.execute([img])

    print(*r)
