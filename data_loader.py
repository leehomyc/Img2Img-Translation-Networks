"""The data loader for model training."""
import os

import tensorflow as tf

import config
import model

# -----------------------------------------------------------------------------

_INPUT_SIZE = 256

# -----------------------------------------------------------------------------


def _load_samples(csv_name):
    """Read and decode a pair of images.

    Args:
        csv_name: A string that describes the name of the csv file.

    Returns:
        image_decoded_a: A tensor as the decoded first image.
        image_decoded_b: A tensor as the decoded second image.
        filename_i: A tensor as the name of the first image.
        filename_j: A tensor as the name of the second image.
    """
    filename_queue = tf.train.string_input_producer(
        [csv_name])

    reader = tf.TextLineReader()
    _, csv_filename = reader.read(filename_queue)

    record_defaults = [tf.constant([], dtype=tf.string),
                       tf.constant([], dtype=tf.string)]
    filename_i, filename_j = tf.decode_csv(
        csv_filename, record_defaults=record_defaults)

    file_contents_i = tf.read_file(filename_i)
    file_contents_j = tf.read_file(filename_j)

    # syn image is png, celebA is jpg.
    image_decoded_a = tf.image.decode_png(
        file_contents_i, channels=model.IMG_CHANNELS, dtype=tf.uint8)
    image_decoded_b = tf.image.decode_jpeg(
        file_contents_j, channels=model.IMG_CHANNELS)

    return image_decoded_a, image_decoded_b, filename_i, filename_j


def load_data(split_name, shuffle=False):
    """Load and pre-process images of the two domains.

    Args:
        split_name: A string as the name of the csv file.
        shuffle: A boolen variable to indicate whether to shuffle the dataset.

    Returns:
        A dictionary of four tensors "image_i", "image_j", "filename_i",
            "filename_j".
    """
    csv_file = os.path.join(config._ROOT_DIR, '{}.csv'.format(split_name))

    image_i, image_j, filename_i, filename_j = _load_samples(csv_file)

    image_i = tf.image.resize_images(
        image_i,
        [_INPUT_SIZE, _INPUT_SIZE],
    )
    image_j = tf.image.resize_images(
        image_j,
        [_INPUT_SIZE, _INPUT_SIZE],
    )

    image_i = image_i / 127.5 - 1
    image_j = image_j / 127.5 - 1

    if shuffle is True:
        images_i, images_j, filenames_i, filenames_j = tf.train.shuffle_batch(
            [image_i, image_j, filename_i, filename_j], 1, 5000, 100)
    else:
        images_i, images_j, filenames_i, filenames_j = tf.train.batch(
            [image_i, image_j, filename_i, filename_j], 1)

    return {
        'images_i': images_i,
        'images_j': images_j,
        'filenames_i': filenames_i,
        'filenames_j': filenames_j,
    }
