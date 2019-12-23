import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as F
import time
from threading import Thread, Lock
import traceback
import logging
from src.image_processor.image_processor import ImageProcessor
from src.utils.download_gdrive import download_gdrive


class ModelWorker(ImageProcessor):

    def __init__(
            self,
            traced_model_path=None,
            image_shape=(416, 416),
            id=0
    ):
        super(ModelWorker, self).__init__()
        self.id = id
        self.mutex = Lock()
        self.traced_model_path = traced_model_path if traced_model_path is not None else "traced_model.pt"
        if not os.path.isfile(self.traced_model_path):
            logging.info(f"{self.__class__.__name__}: {self.id} download traced model params")
            self.traced_file_id = "10xFg7qXLtJ3Oc6rQyOlumkoaOU1U4PiU"
            download_gdrive(self.traced_file_id, self.traced_model_path)

        logging.info(f"{self.__class__.__name__}: {self.id} load traced model")
        self.traced_model = torch.jit.load(self.traced_model_path)
        self.image_shape = image_shape
        logging.info(f"{self.__class__.__name__}: {self.id} model ready")

    def __call__(self, image_data: np.ndarray):
        """Threadsafe call function"""
        self.mutex.acquire(False)
        logging.info(f"{self.__class__.__name__}: {self.id} called")
        try:
            # preprocess image_data
            image = image_data.copy()
            if image.ndim == 2:
                # grayscale image
                image = image[:, :, np.newaxis]
                image = np.repeat(image, repeats=3, axis=2)
            elif image.ndim == 3 and image.shape[2] == 4:
                # image with alpha channel
                image = image[:, :, 0:2]

            im_h, im_w = image.shape[:2]
            if im_h > im_w:
                pad_top = pad_down = 0
                pad_left = (im_h - im_w) // 2
                pad_right = im_h - (im_w + pad_left)
            else:
                pad_top = (im_w - im_h) // 2
                pad_down = im_w - (im_h + pad_top)
                pad_left = pad_right = 0
            paddings = (pad_left, pad_top, pad_right, pad_down)

            image = Image.fromarray(image)
            image = F.pad(image, padding=paddings, fill=128)
            image = F.resize(image, self.image_shape)
            image = F.to_tensor(image)
            image = F.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            image = image.unsqueeze(0)

            # predict mask_data
            predicted_mask = self.traced_model(image)

            # post_process mask_data
            predicted_mask = predicted_mask.squeeze()[1]
            predicted_mask = F.to_pil_image(predicted_mask)
            predicted_mask = F.resize(
                predicted_mask,
                (im_h + pad_top + pad_down,
                 im_w + pad_left + pad_right)
            )
            predicted_mask = F.crop(
                predicted_mask,
                i=pad_top,
                j=pad_left,
                h=im_h,
                w=im_w
            )
            predicted_mask = np.array(predicted_mask)
            predicted_mask[predicted_mask > 128] = 255
            predicted_mask[predicted_mask < 255] = 0
            mask_data = predicted_mask
            return image_data, mask_data
        except Exception:
            logging.info(f"{self.__class__.__name__}: {traceback.format_exc()}")
            return image_data, np.zeros(image_data.shape[:2])
        finally:
            self.mutex.release()


class ModelImageProcessor(ImageProcessor):
    def __init__(self, traced_model_path, image_shape=(416, 416), num_workers=1):
        super(ModelImageProcessor, self).__init__()
        self.traced_model_path = traced_model_path
        self.image_shape = image_shape
        self.num_workers = num_workers
        self.workers = [
            ModelWorker(
                traced_model_path=traced_model_path,
                image_shape=self.image_shape,
                id=i
            )
            for i in range(self.num_workers)
        ]

    def __call__(self, image_data: np.ndarray):
        while True:
            for worker in self.workers:
                successfully_acquired = worker.mutex.acquire(False)
                if successfully_acquired:
                    out = worker(image_data=image_data)
                    return out
                else:
                    time.sleep(0.1)


def test_model_worker():
    model_processor = ModelWorker(
        traced_model_path='traced_model.pt',
        image_shape=(416, 416)
    )
    import skimage.io as skio
    image = skio.imread("image_test.png")

    image, mask = model_processor(image)
    skio.imsave("mask_test_0.png", mask)


def test_model_image_processor():

    model_processor = ModelImageProcessor(
        traced_model_path='traced_model.pt',
        image_shape=(416, 416),
        num_workers=2
    )
    import skimage.io as skio
    image = skio.imread("image_test.png")
    image, mask1 = model_processor(image)
    image, mask2 = model_processor(image)

    skio.imsave("mask_test_1.png", mask1)
    skio.imsave("mask_test_2.png", mask2)


if __name__ == '__main__':
    test_model_worker()
    test_model_image_processor()
