import tensorflow as tf
import preprocessing
import lib


def get_model(input, n_classes, in_training_mode):

    input_layer = tf.convert_to_tensor(input, dtype=tf.float32)

    inputs_size = tf.shape(input_layer)[1:3]
    # block1
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='block1_conv1')
    conv1 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='block1_conv2')
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='block1_pool')

    # block2
    conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='block2_conv1')
    conv2 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='block2_conv2')
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='block2_pool')

    # block3
    conv3 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='block3_conv1')
    conv3 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='block3_conv2')
    conv3 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='block3_conv3')
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2, name='block3_pool')
    # print(pool3)
    # block4
    conv4 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='block4_conv1')
    conv4 = tf.layers.conv2d(inputs=conv4, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='block4_conv2')
    conv4 = tf.layers.conv2d(inputs=conv4, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='block4_conv3')
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2, name='block4_pool')

    # Section attached to the model to load real weights but not used
    conv5 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='block5_conv1')
    conv5 = tf.layers.conv2d(inputs=conv5, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='block5_conv2')
    conv5 = tf.layers.conv2d(inputs=conv5, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='block5_conv3')
    pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2, name='block5_pool')

    # mid
    mid = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    mid = tf.keras.layers.BatchNormalization()(mid, training=in_training_mode)

    # decoder
    up1 = tf.layers.conv2d_transpose(mid, filters=256, strides=2, kernel_size=[3, 3], padding='same', name='up1_upsamp')
    up1 = tf.concat([up1, pool3], axis=3, name='up1_concat')
    up1 = tf.layers.conv2d(inputs=up1, filters=256, kernel_size=[3, 3], padding='same', name='up1_conv1')
    up1 = tf.keras.layers.BatchNormalization()(up1, training=in_training_mode)

    up2 = tf.layers.conv2d_transpose(up1, filters=128, strides=2, kernel_size=[3, 3], padding='same', name='up2_upsamp')
    up2 = tf.concat([up2, pool2], axis=3, name='up2_concat')
    up2 = tf.layers.conv2d(inputs=up2, filters=128, kernel_size=[3, 3], padding='same', name='up2_conv')
    up2 = tf.keras.layers.BatchNormalization()(up2, training=in_training_mode)

    up3 = tf.layers.conv2d_transpose(up2, filters=64, strides=2, kernel_size=[3, 3], padding='same', name='up3_upsamp')
    up3 = tf.concat([up3, pool1], axis=3, name='up3_concat')
    up3 = tf.layers.conv2d(inputs=up3, filters=64, kernel_size=[3, 3], padding='same', name='up3_conv')
    up3 = tf.keras.layers.BatchNormalization()(up3, training=in_training_mode)

    up4 = tf.layers.conv2d_transpose(up3, filters=64, strides=2, kernel_size=[3, 3], padding='same', name='up4_upsamp')
    up4 = tf.layers.conv2d(inputs=up4, filters=64, kernel_size=[3, 3], padding='same', name='up4_conv1')
    up4 = tf.layers.conv2d(inputs=up4, filters=64, kernel_size=[3, 3], padding='same', name='up4_conv2')
    up4 = tf.keras.layers.BatchNormalization()(up4, training=in_training_mode)

    final = tf.layers.conv2d(inputs=up4, filters=n_classes, kernel_size=[1, 1], padding="same", activation=None, name='final_1x1_conv')
    logits = tf.image.resize_bilinear(final, inputs_size, name='bilinear_logits')

    return logits


def model_input_fn(features, labels, mode, params):

    # logits fort he model
    logits = get_model(features, params['n_classes'], mode == tf.estimator.ModeKeys.TRAIN)

    # predicted classes using argmax on logits and expanding last dimension
    pred_classes = tf.expand_dims(tf.argmax(logits, axis=3, output_type=tf.int32), 3)

    # decoded labels from model's predictions
    predictions_decoded_labels = tf.py_func(preprocessing.decode_labels,
                                     [pred_classes, params['batch_size'], params['n_classes']],
                                     tf.uint8)
    # prediction dic to return when mode=PREDICT
    predictions = {
        'probabilities': tf.nn.softmax(logits=logits),
        'images': predictions_decoded_labels,
        'classes': pred_classes
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            predictions=predictions,
            mode=mode
        )


    # Images
    images = tf.cast(features, tf.uint8)

    gt_decoded_labels = tf.py_func(preprocessing.decode_labels,
                                   [labels, params['batch_size'], params['n_classes']], tf.uint8)

    # remove the dimention of size one in labels
    labels = tf.squeeze(labels, axis=3)

    logits_by_num_classes = tf.reshape(logits, [-1, params['n_classes']])

    # flattening the labels to select the valid ones
    labels_flat = tf.reshape(labels, [-1, ])
    # choosing the indices of the available categories (white pixels discarded)
    indices = tf.to_int32(labels_flat < params['n_classes'])
    # partitioning the tensors in the using valid indices
    valid_logits = tf.dynamic_partition(logits_by_num_classes, indices, num_partitions=2)[1]
    valid_labels = tf.dynamic_partition(labels_flat, indices, num_partitions=2)[1]

    pred_flat = tf.reshape(pred_classes, [-1, ])
    valid_pred = tf.dynamic_partition(pred_flat, indices, num_partitions=2)[1]

    # cross entropy loss
    loss = tf.losses.sparse_softmax_cross_entropy(logits=valid_logits, labels=valid_labels)

    # make a copy of cross_entropy for logs
    tf.identity(loss, name='cross_entropy')
    tf.summary.scalar('cross_entropy', loss)

    # accuracy
    accuracy = tf.metrics.accuracy(valid_labels, valid_pred)
    tf.identity(accuracy[1], name='accuracy')
    tf.summary.scalar('accuracy', accuracy[1])

    # mean of IoU
    mean_IoU = tf.metrics.mean_iou(valid_labels, valid_pred, params['n_classes'])
    train_mean_iou = lib.compute_mean_iou(mean_IoU[1], params)
    tf.identity(train_mean_iou, name='mean_IoU')
    tf.summary.scalar('mean_IoU', train_mean_iou)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Show images, correct prediction, current prediction
        tf.summary.image('Images', tf.concat(axis=2, values=[images, gt_decoded_labels, predictions_decoded_labels]),
                         max_outputs=lib._MAX_IMAGES_TO_SHOW)

        # optimizer (Adam)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08,
                                           use_locking=False)

        training_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        # return tf.estimator.EstimatorSpec(train_op=training_op, loss=loss, mode=mode)
    else:
        training_op = None

    metrics = {
        'accuracy': accuracy,
        'mean_iou': mean_IoU
    }

    confusion_matrix = tf.confusion_matrix(valid_labels, valid_pred, params['n_classes'])
    predictions['valid_preds'] = valid_pred
    predictions['valid_labels'] = valid_labels
    predictions['confusion_matrix'] = confusion_matrix

    return tf.estimator.EstimatorSpec(
        predictions=predictions,
        train_op=training_op,
        loss=loss,
        mode=mode,
        eval_metric_ops=metrics)

