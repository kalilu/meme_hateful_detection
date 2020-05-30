import sys
import os
PATH_CURRENT = '/home/jupyter/meme_hateful_detection'

PATH_CODE = f'{PATH_CURRENT}/src'
PATH_DATA = f'{PATH_CURRENT}/data'
PATH_MODEL = f'{PATH_CURRENT}/models'
PATH_NOTEBOOKS = f'{PATH_CURRENT}/notebooks'
PATH_CONFIG = f'{PATH_CURRENT}/configs'

PATH_DEV_IMAGES = f'{PATH_CURRENT}/data/raw/flickr8k_images'
PATH_TEST_IMAGES = f'{PATH_CURRENT}/data/raw/test_images'
PATH_TRAIN_IMAGES = f'{PATH_CURRENT}/data/raw/flickr8k_images'
PATH_TRAIN_IMAGES_TEXT = f'{PATH_CURRENT}/data/raw/flickr8k_images_text'
PATH_TRAIN_MEMES = f'{PATH_CURRENT}/data/raw/facebook_memes'
PATH_MEMES_DATASET = f'{PATH_CURRENT}/data/raw/meme_dataset'

PATH_IMAGE_GENERATOR = f'{PATH_CODE}/cap_generator'

# module_path = PATH_CURRENT
if PATH_CURRENT not in sys.path:
    sys.path.append(PATH_CURRENT)
if PATH_CODE not in sys.path:
    sys.path.append(PATH_CODE)
if PATH_MODEL not in sys.path:
    sys.path.append(PATH_MODEL)
if PATH_TRAIN_IMAGES_TEXT not in sys.path:
    sys.path.append(PATH_TRAIN_IMAGES_TEXT)
if PATH_TEST_IMAGES not in sys.path:
    sys.path.append(PATH_TEST_IMAGES)
if PATH_IMAGE_GENERATOR not in sys.path:
    sys.path.append(PATH_IMAGE_GENERATOR)
    
# import meme_hateful_detection.processors.processors
# import meme_hateful_detection.models.concat_vl
