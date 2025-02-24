import os
import sys
import torch
from pathlib import Path
from typing import List, Optional, Tuple, Iterable, Union, Any

import numpy as np

from torchvision import transforms as tf

class SUT:
    def __init__(self, sut_folder: Optional[Path] = None):
        if sut_folder is None:
            sut_folder = Path(__file__).parent

        self.device = 'cuda'
        self.net = torch.jit.load(sut_folder / "model_script.pth", map_location=self.device)
        self.net.eval()
        self.net.to(self.device)

        self.transform = tf.Compose([
            tf.ToPILImage(),
            tf.Resize((32, 32)),
            # tf.Grayscale(num_output_channels=3),
            tf.RandomEqualize(p=1.0),
            tf.ToTensor(),
            # tf.Normalize((0.5317,), (0.3174,))
        ])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...

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
            probabilities, classes = torch.softmax(outputs, dim=1).max(dim=1)

        return probabilities, classes, outputs


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    img = plt.imread(sys.argv[1])

    with SUT(Path(sys.argv[2])) as s:
        r = s.execute([img])

    print(*r)
