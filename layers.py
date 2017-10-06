import tensorflow as tf


def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):
    with tf.variable_scope(name):
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)
        else:
            return tf.maximum(x, leak * x)


def instance_norm(x):
    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(
                                    mean=1.0, stddev=0.02,
        ))
        offset = tf.get_variable(
            'offset', [x.get_shape()[-1]],
            initializer=tf.constant_initializer(0.0),
        )
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

        return out


def general_conv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02,
                   padding="VALID", name="conv2d", do_norm=True, do_relu=True,
                   relufactor=0):
    with tf.variable_scope(name):

        conv = tf.contrib.layers.conv2d(
            inputconv, o_d, f_w, s_w, padding,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=stddev,
            ),
            biases_initializer=tf.constant_initializer(0.0),
        )
        if do_norm:
            conv = instance_norm(conv)

        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv


def general_deconv2d(inputconv, outshape, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1,
                     stddev=0.02, padding="VALID", name="deconv2d",
                     do_norm=True, do_relu=True, relufactor=0):
    with tf.variable_scope(name):

        conv = tf.contrib.layers.conv2d_transpose(
            inputconv, o_d, [f_h, f_w],
            [s_h, s_w], padding,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            biases_initializer=tf.constant_initializer(0.0),
        )

        if do_norm:
            conv = instance_norm(conv)
            # conv = tf.contrib.layers.batch_norm(conv, decay=0.9,
            # updates_collections=None, epsilon=1e-5, scale=True,
            # scope="batch_norm")

        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv

# The following layers are for pix2pix


def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable(
            "offset", [channels], dtype=tf.float32,
            initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [
                                channels], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(
            input, mean, variance, offset, scale,
            variance_epsilon=variance_epsilon)
        return normalized


def p2p_lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))  # noqa
        # [batch, in_height, in_width, in_channels],
        # [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [
                              1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [
                            1, stride, stride, 1], padding="VALID")
        return conv


def deconv(batch_input, out_channels, name="deconv"):
    with tf.variable_scope(name):
        batch, in_height, in_width, in_channels = [
            int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))  # noqa
        conv = tf.nn.conv2d_transpose(batch_input, filter, [
                                      batch, in_height * 2, in_width * 2,
                                      out_channels], [1, 2, 2, 1], padding="SAME")  # noqa
        return conv
