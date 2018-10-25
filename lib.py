import tensorflow as tf
import os
import preprocessing

train_record = 'voc_train.record'
validation_record = 'voc_val.record'


# Constants
records_dir = 'records/'
model_dir = 'model/vgg_unet_model/'

_NUM_CLASSES = 21
_HEIGHT = 512
_WIDTH = 512
_MIN_SCALE = 0.5
_MAX_SCALE = 2.0
_IGNORE_LABEL = 255

_DEPTH = 3
_BATCH_SIZE = 2
_EPOCHS = 1
_EPOCHS_PER_EVAL = 1
_MAX_IMAGES_TO_SHOW = 5
_NUM_IMAGES = {
    'train': 10582,
    'validation': 1114,
}


def get_record(is_training, data_dir):
    if is_training:
        return [os.path.join(data_dir, train_record)]
    else:
        return [os.path.join(data_dir, validation_record)]


def parse_record(raw_record):
    keys_to_features = {
        'image/height':
            tf.FixedLenFeature((), tf.int64),
        'image/width':
            tf.FixedLenFeature((), tf.int64),
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'label/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'label/format':
            tf.FixedLenFeature((), tf.string, default_value='png'),
    }

    parsed = tf.parse_single_example(raw_record, keys_to_features)

    # height = tf.cast(parsed['image/height'], tf.int32)
    # width = tf.cast(parsed['image/width'], tf.int32)

    image = tf.image.decode_image(
        tf.reshape(parsed['image/encoded'], shape=[]), _DEPTH)
    image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
    image.set_shape([None, None, 3])

    label = tf.image.decode_image(
        tf.reshape(parsed['label/encoded'], shape=[]), 1)
    label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
    label.set_shape([None, None, 1])

    return image, label

def preprocess_image(image, label, is_training):
    """Preprocess a single image of layout [height, width, depth]."""
    if is_training:
        # Randomly scale the image and label.
        # image, label = preprocessing.random_rescale_image_and_label(
        #     image, label, _MIN_SCALE, _MAX_SCALE)

        # Randomly crop or pad a [_HEIGHT, _WIDTH] section of the image and label.
        image, label = preprocessing.random_crop_or_pad_image_and_label(
            image, label, _HEIGHT, _WIDTH, _IGNORE_LABEL)

        # Randomly flip the image and label horizontally.
        image, label = preprocessing.random_flip_left_right_image_and_label(
            image, label)

        image.set_shape([_HEIGHT, _WIDTH, 3])
        label.set_shape([_HEIGHT, _WIDTH, 1])

    image = preprocessing.mean_image_subtraction(image)

    return image, label


def compute_mean_iou(total_cm, params, name='mean_iou'):
    """Compute the mean intersection-over-union via the confusion matrix."""
    sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
    sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
    cm_diag = tf.to_float(tf.diag_part(total_cm))
    denominator = sum_over_row + sum_over_col - cm_diag

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = tf.reduce_sum(tf.cast(
        tf.not_equal(denominator, 0), dtype=tf.float32))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = tf.where(
        tf.greater(denominator, 0),
        denominator,
        tf.ones_like(denominator))
    iou = tf.div(cm_diag, denominator)

    for i in range(params['n_classes']):
        tf.identity(iou[i], name='train_iou_class{}'.format(i))
        tf.summary.scalar('train_iou_class{}'.format(i), iou[i])

    # If the number of valid entries is 0 (no classes) we return 0.
    result = tf.where(
        tf.greater(num_valid_entries, 0),
        tf.reduce_sum(iou, name=name) / num_valid_entries,
        0)
    return result


def input_fn_from_tfrecords(is_training, data_dir, batch_size, epochs):
    file_name = get_record(is_training, data_dir)
    dataset = tf.data.TFRecordDataset(file_name)

    if is_training:
        dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])

    dataset = dataset.map(parse_record)
    dataset = dataset.map(
        lambda image, label: preprocess_image(image, label, is_training))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.repeat(epochs)
    iterator = dataset.make_one_shot_iterator()
    img, ann = iterator.get_next()

    return img, ann


def input_fn_from_folder(image_filenames, label_filenames=None, batch_size=1):

    def _parse_function(filename, is_label):
        if not is_label:
            image_filename, label_filename = filename, None
        else:
            image_filename, label_filename = filename

        image_string = tf.read_file(image_filename)
        image = tf.image.decode_image(image_string)
        image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
        image.set_shape([None, None, 3])

        image = preprocessing.mean_image_subtraction(image)

        if not is_label:
            return image
        else:
            label_string = tf.read_file(label_filename)
            label = tf.image.decode_image(label_string)
            label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
            label.set_shape([None, None, 1])

            return image, label

    if label_filenames is None:
        input_filenames = image_filenames
    else:
        input_filenames = (image_filenames, label_filenames)

    dataset = tf.data.Dataset.from_tensor_slices(input_filenames)
    if label_filenames is None:
        dataset = dataset.map(lambda x: _parse_function(x, False))
    else:
        dataset = dataset.map(lambda x, y: _parse_function((x, y), True))
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()

    if label_filenames is None:
        images = iterator.get_next()
        labels = None
    else:
        images, labels = iterator.get_next()

    return images, labels

