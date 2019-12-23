import numpy as np
from src.sticker.utils import crop_image_by_mask
from src.image_processor.image_processor import ImageProcessor
import skimage.io as skio


class StickerImageProcessor(ImageProcessor):

    def __init__(
            self,
            erode_thickness=4,
            edge_thickness=5,
            edge_color=(255, 255, 255, 255)
    ):
        self.erode_thickness = erode_thickness
        self.edge_thickness = edge_thickness
        self.edge_color = edge_color

    def __call__(self, image_data: np.ndarray, mask_data: np.ndarray):
        """
        image_data, mask_data and make tgs as output

        :param image_data:
        :param mask_data:
        :return:
        """
        image_processed = crop_image_by_mask(
            image=image_data,
            mask=mask_data,
            erode_thickness=self.erode_thickness,
            edge_thickness=self.edge_thickness,
            edge_color=self.edge_color
        )
        return image_processed


if __name__ == '__main__':
    sticker_processor = StickerImageProcessor(
        erode_thickness=3,
        edge_thickness=10,
        edge_color=(255, 255, 255, 255)
    )
    import skimage.io as skio
    image = skio.imread("image_test.png")
    mask = skio.imread("mask_test.png")
    result = sticker_processor(image, mask)
    skio.imsave("sticker_test.png", result)
