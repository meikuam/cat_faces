import torch


def maybe_download(model_name, model_url, model_dir=None, map_location=None):
    import os, sys
    from six.moves import urllib
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = '{}.pth.tar'.format(model_name)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        url = model_url
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urllib.request.urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)


def imread(url, proxy_url=None, timeout=5):
    """
    load image with specified user_agent
    :param url:
    :param proxy_url: url of proxy to use, example: https://ip.port
    :param timeout: connection timeout
    :return: np.ndarray image
    """
    import numpy as np
    import requests
    from PIL import Image
    import io

    if proxy_url is not None:
        proxy_url = {
            proxy_url.split(':')[0]: proxy_url
        }

    r = requests.get(
        url,
        proxies=proxy_url,
        headers={
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36'
        },
        timeout=timeout
    )
    image = Image.open(
        io.BytesIO(
            r.content
        )
    )
    return np.array(image)
