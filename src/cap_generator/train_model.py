from cap_generator import load_data as ld
from cap_generator import generate_model as gen
from keras.callbacks import ModelCheckpoint
from pickle import dump
import os

import sys
import os
PATH_CURRENT = '/home/jupyter/meme_hateful_detection'


PATH_CODE = f'{PATH_CURRENT}/src'
PATH_DATA = f'{PATH_CURRENT}/data'
PATH_MODEL = f'{PATH_CURRENT}/models'
PATH_NOTEBOOKS = f'{PATH_CURRENT}/notebooks'

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
    
def train_model(weight = None, epochs = 10):
  # load dataset
  data = ld.prepare_dataset('train')
  train_features, train_descriptions = data[0]
  test_features, test_descriptions = data[1]

  # prepare tokenizer
  tokenizer = gen.create_tokenizer(train_descriptions)
  # save the tokenizer
  dump(tokenizer, open(f'{PATH_MODEL}/tokenizer.pkl', 'wb'))
  # index_word dict
  index_word = {v: k for k, v in tokenizer.word_index.items()}
  # save dict
  dump(index_word, open(f'{PATH_MODEL}/index_word.pkl', 'wb'))

  vocab_size = len(tokenizer.word_index) + 1
  print('Vocabulary Size: %d' % vocab_size)

  # determine the maximum sequence length
  max_length = gen.max_length(train_descriptions)
  print('Description Length: %d' % max_length)

  # generate model
  model = gen.define_model(vocab_size, max_length)

  # Check if pre-trained weights to be used
  if weight != None:
    model.load_weights(weight)

  # define checkpoint callback
  filepath = PATH_MODEL+'/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                save_best_only=True, mode='min')

  steps = len(train_descriptions)
  val_steps = len(test_descriptions)
  # create the data generator
  train_generator = gen.data_generator(train_descriptions, train_features, tokenizer, max_length)
  val_generator = gen.data_generator(test_descriptions, test_features, tokenizer, max_length)

  # fit model
  model.fit_generator(train_generator, epochs=epochs, steps_per_epoch=steps, verbose=1,
        callbacks=[checkpoint], validation_data=val_generator, validation_steps=val_steps)

  try:
      model.save(f'{PATH_MODEL}/wholeModel.h5', overwrite=True)
      model.save_weights(f'{PATH_MODEL}/weights.h5',overwrite=True)
  except:
      print("Error in saving model.")
  print("Training complete...\n")

if __name__ == '__main__':
    train_model(epochs=20)
