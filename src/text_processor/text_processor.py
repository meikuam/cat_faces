import numpy as np


class TextProcessor:
    """
    Base class for text processors
    """
    def __init__(self):
        super(TextProcessor, self).__init__()

    def __call__(self, *args):
        """
        make some process with text and return answer to text
        :return: response_text
        """
        response_text = ''
        return response_text
