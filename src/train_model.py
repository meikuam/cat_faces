import os
import sys
sys.path.append('.')

import collections
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from catalyst import utils
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import DiceCallback, IouCallback, CriterionCallback, CriterionAggregatorCallback
from catalyst.contrib.criterion import DiceLoss, IoULoss
from catalyst.contrib.optimizers import RAdam, Lookahead

from src.dataset.dataset import PetDataset
from src.model.mobilenet import MBv2, models_urls
from src.utils.helpers import maybe_download
from src.model.model_wrapper import ModelWrapper


if __name__ == "__main__":

    utils.set_global_seed(42)
    utils.prepare_cudnn(deterministic=True)

    image_size = (416, 416)
    batch_size = 8
    num_workers = 0

    dataset = PetDataset(
        data_path="data",
        image_size=image_size,
        use_aug=True,
        to_tensors=True
    )
    print(f"len of dataset: {len(dataset)}")

    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, (train_size, val_size))

    # Catalyst uses normal torch.data.DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )

    # And excpect to get an OrderedDict of loaders
    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader
    ##############################
    model = MBv2(num_classes=2)
    # load pretrained weights
    model_state = model.state_dict()
    bname = 'mbv2_voc'
    key = 'rf_lw' + bname
    url = models_urls[bname]
    state_dict = maybe_download(key, url)

    for key in model_state.keys():
        if key in state_dict:
            if model_state[key].shape != state_dict[key].shape:

                if len(model_state[key].shape) != len(state_dict[key].shape):
                    print('uncapable key:', key)
                elif len(model_state[key].shape) == 1:
                    shape_copy = [
                        min(model_state[key].shape[0], state_dict[key].shape[0]),
                    ]
                    model_state[key][:shape_copy[0]] = state_dict[key][:shape_copy[0]]
                elif len(model_state[key].shape) == 2:
                    shape_copy = [
                        min(model_state[key].shape[0], state_dict[key].shape[0]),
                        min(model_state[key].shape[1], state_dict[key].shape[1]),
                    ]
                    model_state[key][
                    :shape_copy[0],
                    :shape_copy[1]] = state_dict[key][
                                      :shape_copy[0],
                                      :shape_copy[1]]

                elif len(model_state[key].shape) == 3:
                    shape_copy = [
                        min(model_state[key].shape[0], state_dict[key].shape[0]),
                        min(model_state[key].shape[1], state_dict[key].shape[1]),
                        min(model_state[key].shape[2], state_dict[key].shape[2]),
                    ]
                    model_state[key][
                    :shape_copy[0],
                    :shape_copy[1],
                    :shape_copy[2]] = state_dict[key][
                                      :shape_copy[0],
                                      :shape_copy[1],
                                      :shape_copy[2]]
                elif len(model_state[key].shape) == 4:

                    shape_copy = [
                        min(model_state[key].shape[0], state_dict[key].shape[0]),
                        min(model_state[key].shape[1], state_dict[key].shape[1]),
                        min(model_state[key].shape[2], state_dict[key].shape[2]),
                        min(model_state[key].shape[3], state_dict[key].shape[3]),
                    ]
                    model_state[key][
                    :shape_copy[0],
                    :shape_copy[1],
                    :shape_copy[2],
                    :shape_copy[3]] = state_dict[key][
                                      :shape_copy[0],
                                      :shape_copy[1],
                                      :shape_copy[2],
                                      :shape_copy[3]]
            else:
                model_state[key] = state_dict[key]

    model.load_state_dict(model_state, strict=False)

    model_wrapper = ModelWrapper(model)
    ##############################
    learning_rate = 0.001

    # This function removes weight_decay for biases and applies our layerwise_params
    model_params = utils.process_model_params(model_wrapper)

    # Catalyst has new SOTA optimizers out of box
    base_optimizer = RAdam(model_params, lr=learning_rate, weight_decay=0.0003)
    optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2)


    #########################

    num_epochs = 50
    logdir = "logs_pretrain/segmentation"
    resume_path = f"{logdir}/checkpoints/last_full.pth"
    if not os.path.isfile(resume_path):
        resume_path = None
    device = utils.get_device()
    print(f"device: {device}")

    # by default SupervisedRunner uses "features" and "targets",
    # in our case we get "image" and "mask" keys in dataset __getitem__
    runner = SupervisedRunner(device=device, input_key="image", input_target_key="mask")


    # we have multiple criterions
    criterion = {
        "dice": DiceLoss(),
        "iou": IoULoss(),
        "bce": nn.BCEWithLogitsLoss()
    }
    print(f"python -m tensorboard.main --port 6006 --bind_all --logdir={logdir}")
    runner.train(
        model=model_wrapper,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        # our dataloaders
        loaders=loaders,
        callbacks=[
            # Each criterion is calculated separately.
            CriterionCallback(
                input_key="mask",
                prefix="loss_dice",
                criterion_key="dice"
            ),
            CriterionCallback(
                input_key="mask",
                prefix="loss_iou",
                criterion_key="iou"
            ),
            CriterionCallback(
                input_key="mask",
                prefix="loss_bce",
                criterion_key="bce"
            ),
            # And only then we aggregate everything into one loss.
            CriterionAggregatorCallback(
                prefix="loss",
                loss_aggregate_fn="weighted_sum",  # can be "sum", "weighted_sum" or "mean"
                # because we want weighted sum, we need to add scale for each loss
                loss_keys={"loss_dice": 1.0, "loss_iou": 1.0, "loss_bce": 0.8},
            ),
            # metrics
            DiceCallback(input_key="mask"),
            IouCallback(input_key="mask"),
        ],
        # path to save logs
        logdir=logdir,
        # path to resume training if available
        resume=resume_path,
        num_epochs=num_epochs,
        # save our best checkpoint by IoU metric
        main_metric="iou",
        # IoU needs to be maximized.
        minimize_metric=False,
        # for FP16. It uses the variable from the very first cell
        fp16=None,
        # prints train logs
        verbose=True,
    )
