"""Code for training unsupervised image to image translation networks."""
from datetime import datetime
import os
import random

import click
import numpy as np
from scipy.misc import imsave
import tensorflow as tf
from tensorflow.contrib.metrics.python.ops import metric_ops
from tensorflow.python.ops import math_ops

import config
import data_loader
import losses
import model

slim = tf.contrib.slim


class Img2Img:
    """The Img2Img translation network module."""

    def __init__(self, split_name, params, base_lr=.0002, max_step=200,
                 checkpoint_dir='', network_structure='pix2pix'):
        """Init function of Img2Img.

        Args:
            split_name: The name of the dataset.
            params: The dictionary of hyper-parameters.
            base_lr: The base learning rate for the first 100 epochs.
            max_steps: The number of epochs.
            checkpoint_dir: The path to restore previous checkpoints.
            network_structure: A string as the structure of the network.
                It could be 'pix2pix' or 'resnet'
        """
        self._split_name = split_name

        output_root_dir = config._ROOT_DIR
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._output_dir = os.path.join(output_root_dir, current_time)
        self._images_dir = os.path.join(self._output_dir, 'imgs')
        os.makedirs(self._images_dir, exist_ok=True)

        self._pool_size = 50
        self._num_imgs_to_save = 20
        self._save_every_iterations = 1500

        if checkpoint_dir == '':
            self._to_restore = 0
        else:
            self._to_restore = 1
        self._checkpoint_dir = checkpoint_dir

        self._base_lr = base_lr
        self._max_step = max_step

        # hyper-parameters
        self._cycle_lambda_a = params['cycle_lambda']
        self._cycle_lambda_b = params['cycle_lambda']
        self._rec_lambda_a = params['rec_lambda']
        self._rec_lambda_b = params['rec_lambda']
        self._lsgan_lambda_a = params['lsgan_lambda_a']
        self._lsgan_lambda_b = params['lsgan_lambda_b']
        self._num_separate_layers_g = params['num_separate_layers_g']
        self._num_separate_layers_d = params['num_separate_layers_d']
        self._num_no_skip_layers = params['num_no_skip_layers']
        self._lr_g_mult = params['lr_g_mult']
        self._lr_d_mult = params['lr_d_mult']

        self._network_structure = network_structure

    def model_setup(self):
        """Set up the model to train."""
        self.input_a = tf.placeholder(
            tf.float32, [
                1,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                model.IMG_CHANNELS,
            ], name="input_A")
        self.input_b = tf.placeholder(
            tf.float32, [
                1,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                model.IMG_CHANNELS,
            ], name="input_B")

        self.fake_images_A = np.zeros(
            (self._pool_size, 1, model.IMG_HEIGHT, model.IMG_WIDTH,
             model.IMG_CHANNELS),
        )
        self.fake_images_B = np.zeros(
            (self._pool_size, 1, model.IMG_HEIGHT, model.IMG_WIDTH,
             model.IMG_CHANNELS),
        )

        self.fake_pool_A = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                model.IMG_CHANNELS,
            ], name="fake_pool_A")
        self.fake_pool_B = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                model.IMG_CHANNELS,
            ], name="fake_pool_B")

        self.global_step = slim.get_or_create_global_step()
        self.num_fake_inputs = 0
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="lr")

        inputs = {
            'images_a': self.input_a,
            'images_b': self.input_b,
            'fake_pool_a': self.fake_pool_A,
            'fake_pool_b': self.fake_pool_B,
        }

        outputs = model.get_outputs(
            inputs, variable_scope='img2img',
            num_separate_layers=self._num_separate_layers_g,
            num_separate_layers_d=self._num_separate_layers_d,
            num_no_skip_layers=self._num_no_skip_layers,
            network_structure=self._network_structure)

        self.prob_real_a_is_real = outputs['prob_real_a_is_real']
        self.prob_real_b_is_real = outputs['prob_real_b_is_real']
        self.fake_images_a = outputs['fake_images_a']
        self.fake_images_b = outputs['fake_images_b']
        self.prob_fake_a_is_real = outputs['prob_fake_a_is_real']
        self.prob_fake_b_is_real = outputs['prob_fake_b_is_real']
        self.cycle_images_a = outputs['cycle_images_a']
        self.cycle_images_b = outputs['cycle_images_b']
        self.prob_fake_pool_a_is_real = outputs['prob_fake_pool_a_is_real']
        self.prob_fake_pool_b_is_real = outputs['prob_fake_pool_b_is_real']
        self.ae_images_a = outputs['ae_images_a']
        self.ae_images_b = outputs['ae_images_b']

    def create_summaries(self):
        """Create summary tensor for tensorboard."""
        self.summaries = \
            [tf.summary.scalar("loss/rec_loss_a", self.reconstruction_loss_a),
             tf.summary.scalar("loss/rec_loss_b", self.reconstruction_loss_b),
             tf.summary.scalar("loss/lsgan_loss_fake_a",
                               self.lsgan_loss_fake_a),
             tf.summary.scalar("loss/lsgan_loss_fake_b",
                               self.lsgan_loss_fake_b),
             tf.summary.scalar("loss/cycle_loss_a",
                               self.cycle_consistency_loss_a),
             tf.summary.scalar("loss/cycle_loss_b",
                               self.cycle_consistency_loss_b),
             tf.summary.scalar("total_loss/g_loss", self.g_loss),
             tf.summary.scalar("total_loss/d_A_loss", self.d_loss_A),
             tf.summary.scalar("total_loss/d_B_loss", self.d_loss_B)]

    def compute_losses(self):
        """Compute losses."""
        self.reconstruction_loss_a = losses.reconstruction_loss(
            real_images=self.input_a,
            generated_images=self.ae_images_a)
        self.reconstruction_loss_b = losses.reconstruction_loss(
            real_images=self.input_b,
            generated_images=self.ae_images_b)

        self.lsgan_loss_fake_a = losses.lsgan_loss_generator(
            self.prob_fake_a_is_real)
        self.lsgan_loss_fake_b = losses.lsgan_loss_generator(
            self.prob_fake_b_is_real)

        self.cycle_consistency_loss_a = losses.cycle_consistency_loss(
            real_images=self.input_a, generated_images=self.cycle_images_a)
        self.cycle_consistency_loss_b = losses.cycle_consistency_loss(
            real_images=self.input_b, generated_images=self.cycle_images_b)

        self.g_loss = self._rec_lambda_a * self.reconstruction_loss_a + \
            self._rec_lambda_b * self.reconstruction_loss_b + \
            self._cycle_lambda_a * self.cycle_consistency_loss_a + \
            self._cycle_lambda_b * self.cycle_consistency_loss_b + \
            self._lsgan_lambda_a * self.lsgan_loss_fake_a + \
            self._lsgan_lambda_b * self.lsgan_loss_fake_b

        self.d_loss_A = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_a_is_real,
            prob_fake_is_real=self.prob_fake_pool_a_is_real)
        self.d_loss_B = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_b_is_real,
            prob_fake_is_real=self.prob_fake_pool_b_is_real)

        self.model_vars = tf.trainable_variables()

        d_a_vars = [var for var in self.model_vars if 'd1' in var.name or
                    'd_shared' in var.name]
        d_b_vars = [var for var in self.model_vars if 'd2' in var.name or
                    'd_shared' in var.name]
        g_vars = [var for var in self.model_vars
                  if 'ae1' in var.name or 'ae2' in var.name or
                  'ae_shared' in var.name]

        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)

        self.d_A_trainer = optimizer.minimize(self.d_loss_A, var_list=d_a_vars)
        self.d_B_trainer = optimizer.minimize(self.d_loss_B, var_list=d_b_vars)
        self.g_trainer = optimizer.minimize(self.g_loss, var_list=g_vars)

        self.create_summaries()

    def create_metrics(self):
        """Creating the discriminator accuracy metric.

        For the real image the ground truth labeling is 1. For the fake image
        the ground truth labeling is 0. We use threshold .5 to get predictions
        from labels.
        """
        probs = tf.concat([self.prob_real_a_is_real,
                           self.prob_real_b_is_real,
                           self.prob_fake_pool_a_is_real,
                           self.prob_fake_pool_b_is_real], axis=0)
        predictions = math_ops.to_float(math_ops.greater_equal(probs, .5))

        return metric_ops.streaming_accuracy(
            predictions=predictions, labels=tf.constant([1, 1, 0, 0]))

    def save_images(self, sess, epoch):
        """It saves intermediate image results and html visualization.

        Args:
            sess: The current TF session.
            epoch: An integer that describes the current epoch id.
        """
        names = ['inputA_', 'inputB_', 'fakeA_', 'fakeB_', 'cycA_', 'cycB_']

        with open(os.path.join(self._output_dir, 'epoch_' + str(epoch) +
                               '.html'), 'w') as v_html:
            v_html.write("<p><font size=\"6\">From left to right: Input A, Input B, Fake B, Fake A, Cycle A, Cycle B</font></p>")  # noqa
            for i in range(0, self._num_imgs_to_save):
                print("Saving image {}/{}".format(
                    i, self._num_imgs_to_save))
                inputs = sess.run(self.inputs)
                fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run([
                    self.fake_images_a,
                    self.fake_images_b,
                    self.cycle_images_a,
                    self.cycle_images_b,
                ], feed_dict={
                    self.input_a: inputs['images_i'],
                    self.input_b: inputs['images_j'],
                })

                tensors = [inputs['images_i'], inputs['images_j'],
                           fake_B_temp, fake_A_temp, cyc_A_temp, cyc_B_temp]

                for name, tensor in zip(names, tensors):
                    image_name = name + str(epoch) + "_" + str(i) + ".jpg"
                    imsave(os.path.join(self._images_dir, image_name),
                           ((tensor[0] + 1) * 127.5).astype(np.uint8))
                    v_html.write("<img src=\"" +
                                 os.path.join('imgs', image_name) + "\">")
                v_html.write("<br>")

    def fake_image_pool(self, num_fakes, fake, fake_pool):
        """Saving the generated image to corresponding pool of images.

        It keeps on feeling the pool till it is full and then randomly
        selects an already stored image and replace it with new one.
        """
        if num_fakes < self._pool_size:
            fake_pool[num_fakes] = fake
            return fake
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self._pool_size - 1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else:
                return fake

    def train(self):
        """Training Function."""
        # Load Dataset from the dataset folder

        self.inputs = data_loader.load_data(self._split_name)

        # Build the network
        self.model_setup()

        # Loss function calculations
        self.compute_losses()

        # Create discriminator accuracy metrics
        acc, acc_ops = self.create_metrics()

        # Initializing the global variables
        init = (tf.global_variables_initializer(),
                tf.local_variables_initializer())
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)

            # Restore the model to run the model from last checkpoint
            if self._to_restore:
                chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
                saver.restore(sess, chkpt_fname)

            writer = tf.summary.FileWriter(self._output_dir)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # Training Loop
            for epoch in range(sess.run(self.global_step), self._max_step):
                print("In the epoch {}".format(epoch))
                saver.save(sess, os.path.join(
                    self._output_dir, "img2img"), global_step=epoch)

                # Dealing with the learning rate as per the epoch number
                if epoch < 100:
                    curr_lr = self._base_lr
                else:
                    curr_lr = self._base_lr - \
                        self._base_lr * (epoch - 100) / 100

                self.save_images(sess, epoch)

                for i in range(0, self._save_every_iterations):
                    print("Processing batch {}/{}".format(
                        i, self._save_every_iterations))

                    inputs = sess.run(self.inputs)

                    # Optimize the G network.
                    sess.run(self.g_trainer,
                        feed_dict={self.input_a: inputs['images_i'],
                                   self.input_b: inputs['images_j'],
                                   self.learning_rate: curr_lr * self._lr_g_mult})  # noqa

                    # Get fake images to add into the pool/
                    fake_A_temp, fake_B_temp = sess.run(
                        [self.fake_images_a,
                         self.fake_images_b],
                        feed_dict={self.input_a: inputs['images_i'],
                                   self.input_b: inputs['images_j']})

                    # Fetch fake images from the pool.
                    fake_a_temp1 = self.fake_image_pool(
                        self.num_fake_inputs, fake_A_temp, self.fake_images_A)
                    fake_b_temp1 = self.fake_image_pool(
                        self.num_fake_inputs, fake_B_temp, self.fake_images_B)

                    # Optimizing the D_B network
                    sess.run(self.d_B_trainer,
                             feed_dict={self.input_a: inputs['images_i'],
                                        self.input_b: inputs['images_j'],
                                        self.learning_rate: curr_lr * self._lr_d_mult,  # noqa
                                        self.fake_pool_B: fake_b_temp1})

                    # Optimizing the D_A network
                    sess.run(self.d_A_trainer,
                             feed_dict={self.input_a: inputs['images_i'],
                                        self.input_b: inputs['images_j'],
                                        self.learning_rate: curr_lr * self._lr_d_mult,  # noqa
                                        self.fake_pool_A: fake_a_temp1})

                    # add summaries
                    for sum in self.summaries:
                        summary_str = sess.run(
                            sum,
                            feed_dict={self.input_a: inputs['images_i'],
                                       self.input_b: inputs['images_j'],
                                       self.fake_pool_A: fake_a_temp1,
                                       self.fake_pool_B: fake_b_temp1})
                        writer.add_summary(summary_str, epoch *
                                           self._save_every_iterations + i)

                    writer.flush()

                    # add metrics
                    sess.run([acc, acc_ops],
                             feed_dict={self.input_a: inputs['images_i'],
                                        self.input_b: inputs['images_j'],
                                        self.fake_pool_A: fake_a_temp1,
                                        self.fake_pool_B: fake_b_temp1})
                    accuracy = sess.run(acc)
                    print(
                        'current discriminator accuracy: {}'.format(accuracy))

                    self.num_fake_inputs += 1

                sess.run(tf.assign(self.global_step, epoch + 1))

            coord.request_stop()
            coord.join(threads)
            writer.add_graph(sess.graph)


@click.command()
@click.option('--split_name',
              type=str,
              default='horse_zebra',
              help='The name of the split.')
@click.option('--checkpoint_dir',
              type=click.STRING,
              default='',
              help='The name of the train/test split.')
@click.option('--cycle_lambda',
              type=click.FLOAT,
              default='',
              help='The weight of cycle consistency loss.')
@click.option('--rec_lambda',
              type=click.FLOAT,
              default=None,
              help='The weight of reconstruction loss.')
@click.option('--lsgan_lambda_a',
              type=click.FLOAT,
              default=1,
              help='The weight of lsgan for domain a.')
@click.option('--lsgan_lambda_b',
              type=click.FLOAT,
              default=1,
              help='The weight of lsgan for domain b.')
@click.option('--num_separate_layers_g',
              type=click.INT,
              default=None,
              help='The number of independent layers in G.')
@click.option('--num_separate_layers_d',
              type=click.INT,
              default=None,
              help='The number of independent layers in D.')
@click.option('--num_no_skip_layers',
              type=click.INT,
              default=None,
              help='The number of layers without skip connections.')
@click.option('--lr_g_mult',
              type=click.FLOAT,
              default=1,
              help='The weight of reconstruction loss.')
@click.option('--lr_d_mult',
              type=click.FLOAT,
              default=1,
              help='The weight of reconstruction loss.')
@click.option('--network_structure',
              type=click.STRING,
              default='pix2pix',
              help='The structure of the network.')
def main(split_name, checkpoint_dir, cycle_lambda, rec_lambda,
         lsgan_lambda_a, lsgan_lambda_b, num_separate_layers_g,
         num_separate_layers_d, num_no_skip_layers,
         lr_g_mult, lr_d_mult, network_structure):
    """The main function."""
    params = dict()
    params['cycle_lambda'] = cycle_lambda
    params['rec_lambda'] = rec_lambda
    params['lsgan_lambda_a'] = lsgan_lambda_a
    params['lsgan_lambda_b'] = lsgan_lambda_b
    params['num_separate_layers_g'] = num_separate_layers_g
    params['num_separate_layers_d'] = num_separate_layers_d
    params['num_no_skip_layers'] = num_no_skip_layers
    params['lr_g_mult'] = lr_g_mult
    params['lr_d_mult'] = lr_d_mult

    model = Img2Img(split_name, params, base_lr=.0002, max_step=200,
                    checkpoint_dir='',
                    network_structure=network_structure)

    model.train()


if __name__ == '__main__':
    main()
