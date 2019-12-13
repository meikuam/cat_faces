import sys
sys.path.append('.')

import numpy as np
import torch

from catalyst import utils
from catalyst.dl import SupervisedRunner

from src.model.mobilenet import MBv2
from src.model.model_wrapper import ModelWrapper


if __name__ == "__main__":

    image_size = [1, 3, 416, 416]
    batch = {
        'image': np.random.randn(*image_size)
    }
    # create model
    model = MBv2(num_classes=2)
    model_wrapper = ModelWrapper(model)

    device = torch.device("cpu")
    print(f"device: {device}")

    logdir = "logs/segmentation"
    checkpoint_path = f"{logdir}/checkpoints/best.pth"
    checkpoint = utils.load_checkpoint(f"{logdir}/checkpoints/best.pth")
    model_wrapper.load_state_dict(checkpoint['model_state_dict'])

    # create runner
    runner = SupervisedRunner(device=device, input_key="image", input_target_key="mask")
    # trace model
    # saves to `logdir` and returns a `ScriptModule` class
    traced_script_module = runner.trace(model=model_wrapper, batch=batch, fp16=False)
    traced_script_module.save("traced_model.pt")
