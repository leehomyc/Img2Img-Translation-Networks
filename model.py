"""Code for constructing the model and get the outputs from the model."""
import tensorflow as tf

from . import layers


# -----------------------------------------------------------------------------

slim = tf.contrib.slim

# The number of samples per batch.
BATCH_SIZE = 1

# The height of each image.
IMG_HEIGHT = 256

# The width of each image.
IMG_WIDTH = 256

# The number of color channels per image.
IMG_CHANNELS = 3

# -----------------------------------------------------------------------------


def get_outputs(inputs, variable_scope, num_separate_layers=3,
                num_separate_layers_d=5, num_no_skip_layers=0,
                network_structure='pix2pix'):
    """Get the encoder-decoder and discriminator outputs given the inputs.

    Args:
        inputs: a tensor as the input image.
        variable_scope: a string as the variable scope.
        num_separate_layers: an integer as the number of independent layers
            between two auto-encoders.
        num_no_skip_layers: an integer as the number of layers that do not have
            skip connections in the encoder decoder.
        network_structure: a string to specify the structure of the network.
            It could be "pix2pix" or "resnet".
    Return:
        A list of tensors as the fake images and probabilities.
    """
    images_a = inputs['images_a']
    images_b = inputs['images_b']

    fake_pool_a = inputs['fake_pool_a']
    fake_pool_b = inputs['fake_pool_b']

    if network_structure == 'pix2pix':
        generator = encoder_decoder
    elif network_structure == 'resnet':
        generator = generator_resnet_9blocks
    else:
        raise ValueError("Model [%s] not recognized." % network_structure)

    with tf.variable_scope(variable_scope) as scope:
        ae_images_a = generator(
            images_a, 1, num_separate_layers, num_no_skip_layers)
        ae_images_b = generator(
            images_b, 2, num_separate_layers, num_no_skip_layers)

        fake_images_a = generator(
            images_b, 4, num_separate_layers, num_no_skip_layers)
        fake_images_b = generator(
            images_a, 3, num_separate_layers, num_no_skip_layers)

        cycle_images_b = generator(
            fake_images_a, 3, num_separate_layers, num_no_skip_layers)
        cycle_images_a = generator(
            fake_images_b, 4, num_separate_layers, num_no_skip_layers)

        prob_real_a_is_real = discriminator(images_a, 1, num_separate_layers_d)
        prob_real_b_is_real = discriminator(images_b, 2, num_separate_layers_d)

        scope.reuse_variables()

        prob_fake_a_is_real = discriminator(
            fake_images_a, 1, num_separate_layers_d)
        prob_fake_b_is_real = discriminator(
            fake_images_b, 2, num_separate_layers_d)

        scope.reuse_variables()

        prob_fake_pool_a_is_real = discriminator(
            fake_pool_a, 1, num_separate_layers_d)
        prob_fake_pool_b_is_real = discriminator(
            fake_pool_b, 2, num_separate_layers_d)

    return {
        'prob_real_a_is_real': prob_real_a_is_real,
        'prob_real_b_is_real': prob_real_b_is_real,
        'prob_fake_a_is_real': prob_fake_a_is_real,
        'prob_fake_b_is_real': prob_fake_b_is_real,
        'prob_fake_pool_a_is_real': prob_fake_pool_a_is_real,
        'prob_fake_pool_b_is_real': prob_fake_pool_b_is_real,
        'cycle_images_a': cycle_images_a,
        'cycle_images_b': cycle_images_b,
        'fake_images_a': fake_images_a,
        'fake_images_b': fake_images_b,
        'ae_images_a': ae_images_a,
        'ae_images_b': ae_images_b,
    }


def get_scope_and_reuse_encoder(network_id, layer_id, num_separate_layers):
    """Get the current scope name and whether to reuse parameters.

    Args:
        network_id: an integer as the index of the network.
            If network_id == 1, it is the encoder of the first domain.
            if network_id == 2, it is the encoder of the second domain.
            if network_id == 3, it is the encoder that reuses network 1.
            if network_id == 4, it is the encoder that reuses network 2.
        layer_id: an integer as the index of the layer. The range should be
            (0-8).
        num_separate_layers: an integer as how many layers are separate.
            The rest should be shared between network 1 and 2.

    Return:
        scope: a string as the scope name.
        reuse: a boolean as whether to reuse the parameters.
    """
    if network_id == 1:
        if layer_id < num_separate_layers:
            scope = 'ae1_encoder_{}'.format(layer_id)
        else:
            scope = 'ae_shared_encoder_{}'.format(layer_id)
        reuse = False
    elif network_id == 2:
        if layer_id < num_separate_layers:
            scope = 'ae2_encoder_{}'.format(layer_id)
            reuse = False
        else:
            scope = 'ae_shared_encoder_{}'.format(layer_id)
            reuse = True
    elif network_id == 3:
        if layer_id < num_separate_layers:
            scope = 'ae1_encoder_{}'.format(layer_id)
        else:
            scope = 'ae_shared_encoder_{}'.format(layer_id)
        reuse = True
    elif network_id == 4:
        if layer_id < num_separate_layers:
            scope = 'ae2_encoder_{}'.format(layer_id)
        else:
            scope = 'ae_shared_encoder_{}'.format(layer_id)
        reuse = True
    return scope, reuse


def get_scope_and_reuse_decoder(network_id, layer_id, num_separate_layers):
    """Get the current scope name and whether to reuse parameters.

    Args:
        network_id: an integer as the index of the network.
            If network_id == 1, it is the decoder of the first domain.
            if network_id == 2, it is the decoder of the second domain.
            if network_id == 3, it is the decoder that reuses network 2.
            if network_id == 4, it is the decoder that reuses network 1.
        layer_id: an integer as the index of the layer. The range should be
            (0-8).
        num_separate_layers: an integer as how many layers are separate.
            The rest should be shared between network 1 and 2.

    Return:
        scope: a string as the scope name.
        reuse: a boolean as whether to reuse the parameters.
    """
    if network_id == 1:
        if layer_id < num_separate_layers:
            scope = 'ae1_decoder_{}'.format(layer_id)
            reuse = False
        else:
            scope = 'ae_shared_decoder_{}'.format(layer_id)
            reuse = False
    elif network_id == 2:
        if layer_id < num_separate_layers:
            scope = 'ae2_decoder_{}'.format(layer_id)
            reuse = False
        else:
            scope = 'ae_shared_decoder_{}'.format(layer_id)
            reuse = True
    elif network_id == 3:
        if layer_id < num_separate_layers:
            scope = 'ae2_decoder_{}'.format(layer_id)
        else:
            scope = 'ae_shared_decoder_{}'.format(layer_id)
        reuse = True
    elif network_id == 4:
        if layer_id < num_separate_layers:
            scope = 'ae1_decoder_{}'.format(layer_id)
        else:
            scope = 'ae_shared_decoder_{}'.format(layer_id)
        reuse = True
    return scope, reuse


def get_encoder_layer_specs():
    """Return number of output channels of each encoder layer."""
    return [
        # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        64 * 2,
        # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        64 * 4,
        # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        64 * 8,
        # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        64 * 8,
        # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        64 * 8,
        # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        64 * 8,
        # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        64 * 8,
    ]


def get_decoder_layer_specs():
    """Get number of output channels and dropout ratio in decoder."""
    return [
        # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (64 * 8, 0.5),
        # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 *
        # 2]
        (64 * 8, 0.5),
        # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 *
        # 2]
        (64 * 8, 0.5),
        # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8
        # * 2]
        (64 * 8, 0.0),
        # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 *
        # 2]
        (64 * 4, 0.0),
        # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 *
        # 2]
        (64 * 2, 0.0),
        # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf
        # * 2]
        (64, 0.0),
    ]


def encoder_decoder(inputs, network_id, num_separate_layers,
                    num_no_skip_layers=0):
    """The autoencoder network of the img2img model. We used the network
        architecture from pix2pix.

    Args:
        inputs: a tensor as the input to the encoder decoder.
        network_id: an integer as the index of the network.
            If network_id == 1, it is the encoder_deoder of the first domain.
            if network_id == 2, it is the encoder_deoder of the second domain.
            if network_id == 3, it is the encoder_deoder that reuses network 2.
            if network_id == 4, it is the encoder_deoder that reuses network 1.
        num_separate_layers: an integer as the number of separate layers.
        num_no_skip_layers: an integer as the number of layers without skip
            connection.

    Return:
        A tensor as the output of the encoder_decoder.
    """
    all_layers = []

    scope, reuse = get_scope_and_reuse_encoder(
        network_id, 0, num_separate_layers)
    with tf.variable_scope(scope):
        if reuse is True:
            tf.get_variable_scope().reuse_variables()
        output = layers.conv(inputs, 64, stride=2)
        all_layers.append(output)

    layer_specs = get_encoder_layer_specs()

    total_num_layers = len(layer_specs) + 1
    for i, out_channels in enumerate(layer_specs):
        scope, reuse = get_scope_and_reuse_encoder(
            network_id, (i + 1), num_separate_layers)
        with tf.variable_scope(scope):
            if reuse is True:
                tf.get_variable_scope().reuse_variables()
            rectified = layers.p2p_lrelu(all_layers[-1], 0.2)
            convolved = layers.conv(rectified, out_channels, stride=2)
            output = layers.batchnorm(convolved)
            all_layers.append(output)

    # decoder part
    layer_specs = get_decoder_layer_specs()
    for i, (out_channels, dropout) in enumerate(layer_specs):
        current_layer = total_num_layers - i - 1
        scope, reuse = get_scope_and_reuse_decoder(
            network_id, current_layer, num_separate_layers)
        with tf.variable_scope(scope):
            if reuse is True:
                tf.get_variable_scope().reuse_variables()
            if current_layer == total_num_layers - 1 or \
                    current_layer < num_no_skip_layers:
                input = all_layers[-1]
            else:
                input = tf.concat(
                    [all_layers[-1], all_layers[current_layer]], axis=3)
            rectified = tf.nn.relu(input)
            output = layers.deconv(rectified, out_channels)
            output = layers.batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)
            all_layers.append(output)

    scope, reuse = get_scope_and_reuse_decoder(
        network_id, 0, num_separate_layers)
    with tf.variable_scope(scope):
        if reuse is True:
            tf.get_variable_scope().reuse_variables()
        if 0 < num_no_skip_layers:
            input = tf.concat([all_layers[-1], all_layers[0]], axis=3)
        else:
            input = all_layers[-1]
        rectified = tf.nn.relu(input)
        output = layers.deconv(rectified, 3)
        output = tf.tanh(output)
        all_layers.append(output)

    return all_layers[-1]


def get_scope_and_reuse_conv(network_id):
    """Return the network scope name of conv part given network id.

    We use the ae as name only to make it consistent with pix2pix
    structure but it is not an auto-encoder. For network 1 or
    network 2, the weight is not shared. network 3 shares with network 1
    and network 4 shares with network 2.
    """
    if network_id == 1 or network_id == 2:
        scope = 'ae{}'.format(network_id)
        reuse = False
    elif network_id == 3:
        scope = 'ae1'
        reuse = True
    elif network_id == 4:
        scope = 'ae2'
        reuse = True
    return scope, reuse


def get_scope_and_reuse_resnet(network_id, layer_id, total_num_layers,
                               num_separate_layers):
    """Return the network scope name and reuse flag.

    For network 1 and network 2, if layer_id is smaller than
    num_separate_layers or if layer_id is equal or larger than
    total_num_layers minus num_separate_layers then its not shared.
    For network 3, it shares with network 1 first, and
    then with network 2. For network 4, it shares with network 2
    first, and then with network 1.
    """
    if network_id == 1:
        reuse = False
        if layer_id < num_separate_layers:
            scope = 'ae1'
        elif layer_id >= total_num_layers - num_separate_layers:
            scope = 'ae1'
        else:
            scope = 'ae_shared'
    elif network_id == 2:
        if layer_id < num_separate_layers:
            scope = 'ae2'
            reuse = False
        elif layer_id >= total_num_layers - num_separate_layers:
            scope = 'ae2'
            reuse = False
        else:
            scope = 'ae_shared'
            reuse = True
    elif network_id == 3:
        reuse = True
        if layer_id < num_separate_layers:
            scope = 'ae1'
        elif layer_id >= total_num_layers - num_separate_layers:
            scope = 'ae2'
        else:
            scope = 'ae_shared'
    elif network_id == 4:
        reuse = True
        if layer_id < num_separate_layers:
            scope = 'ae2'
        elif layer_id >= total_num_layers - num_separate_layers:
            scope = 'ae1'
        else:
            scope = 'ae_shared'

    return scope, reuse


def get_scope_and_reuse_deconv(network_id):
    """Return the network scope name given network id.

    For network 1 or network 2, the weight is not shared.
    network 3 shares with network 2 and network 4 shares
    with network 1.
    """
    if network_id == 1 or network_id == 2:
        scope = 'ae{}'.format(network_id)
        reuse = False
    elif network_id == 3:
        scope = 'ae2'
        reuse = True
    elif network_id == 4:
        scope = 'ae1'
        reuse = True
    return scope, reuse


def build_resnet_block(inputres, dim, name="resnet", padding="REFLECT"):
    """Build a single block of resnet.

    :param inputres: inputres
    :param dim: dim
    :param name: name
    :param padding: for tensorflow version use REFLECT; for pytorch version use
     CONSTANT
    :return: a single block of resnet.
    """
    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [
            1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(
            out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(
            out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c2", do_relu=False)

        return tf.nn.relu(out_res + inputres)


def generator_resnet_9blocks(inputs, network_id, num_separate_layers,
                             num_no_skip_layers):
    """Build 9 blocks of ResNet as generator.

    The generator consists of three parts: Conv, ResNet blocks and DeConv.
    Conv and DeConv are not shared. The ResNet blocks are partially shared
    in the middle blocks.

    Args:
        inputs: a tensor as the input image.
        network_id: an integer as the id of the network (1-4).
        num_separate_layers: an integer as the number of separate layers.
        num_no_skip_layers: a dummy variable which is not used.
    """
    fl_ks = 7  # kernel size of the first and last layer
    ks = 3
    padding = "CONSTANT"

    _num_generator_filters = 32

    scope, reuse = get_scope_and_reuse_conv(network_id)
    with tf.variable_scope(scope):
        if reuse is True:
            tf.get_variable_scope().reuse_variables()
        pad_input = tf.pad(
            inputs, [[0, 0], [ks, ks], [ks, ks], [0, 0]], padding)
        o_c1 = layers.general_conv2d(
            pad_input, _num_generator_filters, fl_ks, fl_ks, 1, 1, 0.02, name="c1")  # noqa
        o_c2 = layers.general_conv2d(
            o_c1, _num_generator_filters * 2, ks, ks, 2, 2, 0.02, "SAME", "c2")  # noqa
        o_c3 = layers.general_conv2d(
            o_c2, _num_generator_filters * 4, ks, ks, 2, 2, 0.02, "SAME", "c3")  # noqa

    in_t = o_c3
    for i in range(9):
        scope, reuse = get_scope_and_reuse_resnet(
            network_id, i, 9, num_separate_layers)
        with tf.variable_scope(scope):
            if reuse is True:
                tf.get_variable_scope().reuse_variables()
            out = build_resnet_block(
                in_t, _num_generator_filters * 4, 'r{}'.format(i),
                padding)
            in_t = out

    scope, reuse = get_scope_and_reuse_deconv(network_id)
    with tf.variable_scope(scope):
        if reuse is True:
            tf.get_variable_scope().reuse_variables()
        o_c4 = layers.general_deconv2d(
            out, [BATCH_SIZE, 128, 128, _num_generator_filters *
                  2], _num_generator_filters * 2, ks, ks, 2, 2, 0.02,
            "SAME", "c4")
        o_c5 = layers.general_deconv2d(
            o_c4, [BATCH_SIZE, 256, 256, _num_generator_filters],
            _num_generator_filters, ks, ks, 2, 2, 0.02,
            "SAME", "c5")
        o_c6 = layers.general_conv2d(o_c5, IMG_CHANNELS, fl_ks, fl_ks,
                                     1, 1, 0.02, "SAME", "c6",
                                     do_norm=False, do_relu=False)

        out_gen = tf.nn.tanh(o_c6, "t1")

    return out_gen


def get_discriminator_specs():
    """Return the discriminator specifications at each layer, including number of filters, stride, batch normalization and relu factor."""  # noqa
    return [
        (64, 2, 0, 0.2),
        (64 * 2, 2, 1, 0.2),
        (64 * 4, 2, 1, 0.2),
        (64 * 8, 1, 1, 0.2),
        (1, 1, 0, 0)
    ]


def get_scope_and_reuse_disc(network_id, layer_id, num_separate_layers):
    """Return the scope and reuse flag.

    Args:
        network_index: an integer as the network index.
        layer_id: an integer as the index of the layer.
        num_separate_layers: an integer as how many layers are independent.

    Return:
        scope: a string as the scope.
        reuse: a boolean as the reuse flag.
    """
    if network_id == 1:
        if layer_id < num_separate_layers:
            scope = 'd1_encoder_{}'.format(layer_id)
        else:
            scope = 'd_shared_encoder_{}'.format(layer_id)
        reuse = False
    elif network_id == 2:
        if layer_id < num_separate_layers:
            scope = 'd2_encoder_{}'.format(layer_id)
            reuse = False
        else:
            scope = 'd_shared_encoder_{}'.format(layer_id)
            reuse = True
    return scope, reuse


def discriminator(input, network_index, num_separate_layers):
    """The discriminator.

    Args:
        input: a tensor as the input to the discriminator.
        network_index: an integer as the index of the discriminator.
        num_separate_layers: an integer as the number of independent layers.
    """
    f = 4
    padw = 2

    discriminator_specs = get_discriminator_specs()
    for i, (output_channel, stride, bn, relu) in \
            enumerate(discriminator_specs):
        pad_input = tf.pad(input, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")
        name, reuse = get_scope_and_reuse_disc(
            network_index, i, num_separate_layers)
        with tf.variable_scope(name) as scope:
            if reuse is True:
                scope.reuse_variables()
            input = layers.general_conv2d(
                pad_input, output_channel, f, f, stride, stride, 0.02,
                "VALID", scope, do_norm=bn > 0, do_relu=relu > 0,
                relufactor=relu)
    return input
