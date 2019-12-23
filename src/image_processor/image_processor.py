import numpy as np


class ImageProcessor:
    """
    Base class for image processors
    """
    def __call__(self, image_data: np.ndarray):
        """
        make some process with image and return changed image
        :param image_data: image_data
        :return:
        """
        return image_data


class ComposeImageProcessor(ImageProcessor):

    def __init__(self, processors: list):
        """
        :param processors: list of ImageProcessor instances
        """
        self.processors = processors

    def __call__(self, *args):
        out_data = args
        for processor in self.processors:
            out_data = processor(*out_data)
        return out_data
