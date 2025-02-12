import sys
import numpy as np
from loguru import logger

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


def get_logger(outputfile):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    if outputfile:
        logger.add(outputfile, enqueue=True, format=log_format)
    return logger

def get_sampling_probability(num_epochs, epoch, method, k):
    if method == 'linear':
        p = max(0.1, 1 - epoch / k)
    elif method == 'exp':
        p = k ** epoch
    elif method == 'sigmoid':
        p = k / (k + np.exp(epoch / k))
    elif method == 'cycle':
        p = 0.5 * (1 + np.cos(epoch * 2 * np.pi / k))
    elif method == 'cyclin':
        leni = num_epochs // int(k)
        p = max(0.1, 1 - (epoch % leni) / leni * 2)
    else:
        p = 1  # 默认情况下始终使用真实数据
    return p

def ptb_tokenize(key_to_captions):
    captions_for_image = {}
    for key, caps in key_to_captions.items():
        captions_for_image[key] = []
        for idx, cap in enumerate(caps):
            captions_for_image[key].append(
                {
                    # "image_id": key
                    # "id": idx,
                    "caption": cap
                }
            )
    tokenizer = PTBTokenizer()
    key_to_captions = tokenizer.tokenize(captions_for_image)
    return key_to_captions
