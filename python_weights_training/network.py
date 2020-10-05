import tensorflow as tf
import numpy as np
import os
import random
import cv2
import math

from config import parse_args
from module import model
from data_tfrecord import read_dir, write_tfrecord, read_tfrecord, data_exist

class Network(object):
    def __init__(self, args):
        self.args = args

        CKPT_DIR = args.ckpt_dir

        # Create ckpt directory if it does not exist
        if not os.path.exists(CKPT_DIR):
            os.mkdir(CKPT_DIR)

        # Print Arguments
        args = args.__dict__
        print("Arguments : ")

        for key, value in sorted(args.items()):
            print('\t%15s:\t%s' % (key, value))

        print("Are the arguments correct?")
        input("Press Enter to continue")

    def build(self):

        # Parameters
        DATA_DIR = self.args.data_dir

        LAYER_NUM = self.args.layer_num
        HIDDEN_UNIT = self.args.hidden_unit
        LAMBDA_CTX = self.args.lambda_ctx
        LAMBDA_Y = self.args.lambda_y
        LAMBDA_U = self.args.lambda_u
        LAMBDA_V = self.args.lambda_v
        LR = self.args.lr

        BATCH_SIZE = self.args.batch_size
        CROP_SIZE = self.args.crop_size

        CHANNEL_EPOCH = self.args.channel_epoch
        JOINT_EPOCH = self.args.joint_epoch

        CTX_UP = self.args.ctx_up
        CTX_LEFT = self.args.ctx_left
        CTX_TOTAL = (CTX_LEFT * 2 + 1) * CTX_UP + CTX_LEFT

        tfrecord_name = 'train.tfrecord'

        if not data_exist(DATA_DIR, tfrecord_name):
            img_list = read_dir(DATA_DIR + 'train/')
            write_tfrecord(DATA_DIR, img_list, tfrecord_name)

        input_crop, _, _ = read_tfrecord(DATA_DIR, tfrecord_name, num_epochs=3*CHANNEL_EPOCH+JOINT_EPOCH,
                                        batch_size=4, min_after_dequeue=10, crop_size=CROP_SIZE)

        smooth_data, smooth_label, texture_data, texture_label = self.crop_to_data(input_crop)

        smooth_y_gt = tf.slice(smooth_label, [0, 0], [-1, 1])
        smooth_u_gt = tf.slice(smooth_label, [0, 1], [-1, 1])
        smooth_v_gt = tf.slice(smooth_label, [0, 2], [-1, 1])

        texture_y_gt = tf.slice(texture_label, [0, 0], [-1, 1])
        texture_u_gt = tf.slice(texture_label, [0, 1], [-1, 1])
        texture_v_gt = tf.slice(texture_label, [0, 2], [-1, 1])

        smooth_y_support = tf.slice(smooth_data, [0, 0], [-1, CTX_TOTAL-1])
        smooth_u_support = tf.slice(smooth_data, [0, CTX_TOTAL-1], [-1, CTX_TOTAL-1])
        smooth_v_support = tf.slice(smooth_data, [0, 2*CTX_TOTAL-2], [-1, CTX_TOTAL-1])

        texture_y_support = tf.slice(texture_data, [0, 0], [-1, CTX_TOTAL-1])
        texture_u_support = tf.slice(texture_data, [0, CTX_TOTAL-1], [-1, CTX_TOTAL-1])
        texture_v_support = tf.slice(texture_data, [0, 2*CTX_TOTAL-2], [-1, CTX_TOTAL-1])

        smooth_y_input = smooth_y_support
        smooth_u_input = tf.concat([smooth_y_support, smooth_u_support], axis=1)
        smooth_v_input = tf.concat([smooth_y_support, smooth_v_support], axis=1)

        texture_y_input = texture_y_support
        texture_u_input = tf.concat([texture_y_support, texture_u_support], axis=1)
        texture_v_input = tf.concat([texture_y_support, texture_v_support], axis=1)


        # Smooth Network
        s_out_y, s_hidden_y = model(smooth_y_input, LAYER_NUM, HIDDEN_UNIT, 'smooth_y')

        s_input_f2 = tf.concat([s_hidden_y, smooth_u_input, smooth_y_gt, tf.expand_dims(s_out_y[:,0], axis=1)], axis=1)

        s_out_u, s_hidden_u = model(s_input_f2, LAYER_NUM, HIDDEN_UNIT, 'smooth_u')

        s_input_f3 = tf.concat([s_hidden_u, smooth_v_input, smooth_y_gt, tf.expand_dims(s_out_y[:, 0], axis=1), smooth_u_gt, tf.expand_dims(s_out_u[:, 0], axis=1)], axis=1)

        s_out_v, _, = model(s_input_f3, LAYER_NUM, HIDDEN_UNIT, 'smooth_v')

        # Texture Network
        t_out_y, t_hidden_y = model(texture_y_input, LAYER_NUM, HIDDEN_UNIT, 'texture_y')

        t_input_f2 = tf.concat([t_hidden_y, texture_u_input, texture_y_gt, tf.expand_dims(t_out_y[:,0], axis=1)], axis=1)

        t_out_u, t_hidden_u = model(t_input_f2, LAYER_NUM, HIDDEN_UNIT, 'texture_u')

        t_input_f3 = tf.concat([t_hidden_u, texture_v_input, texture_y_gt, tf.expand_dims(t_out_y[:, 0], axis=1), texture_u_gt, tf.expand_dims(t_out_u[:, 0], axis=1)], axis=1)

        t_out_v, _, = model(t_input_f3, LAYER_NUM, HIDDEN_UNIT, 'texture_v')


        # Smooth Error
        s_pred_y = s_out_y[:, 0]
        s_pred_u = s_out_u[:, 0]
        s_pred_v = s_out_v[:, 0]
        s_ctx_y  = tf.nn.relu(s_out_y[:, 1])
        s_ctx_u  = tf.nn.relu(s_out_u[:, 1])
        s_ctx_v  = tf.nn.relu(s_out_v[:, 1])

        s_predError_y = abs(tf.subtract(s_pred_y, tf.squeeze(smooth_y_gt, axis=1)))
        s_predError_u = abs(tf.subtract(s_pred_u, tf.squeeze(smooth_u_gt, axis=1)))
        s_predError_v = abs(tf.subtract(s_pred_v, tf.squeeze(smooth_v_gt, axis=1)))

        s_loss_pred_y = LAMBDA_Y * tf.reduce_mean(s_predError_y)
        s_loss_pred_u = LAMBDA_U * tf.reduce_mean(s_predError_u)
        s_loss_pred_v = LAMBDA_V * tf.reduce_mean(s_predError_v)

        s_loss_ctx_y = LAMBDA_Y * LAMBDA_CTX * tf.reduce_mean(abs(tf.subtract(s_ctx_y, s_predError_y)))
        s_loss_ctx_u = LAMBDA_U * LAMBDA_CTX * tf.reduce_mean(abs(tf.subtract(s_ctx_u, s_predError_u)))
        s_loss_ctx_v = LAMBDA_V * LAMBDA_CTX * tf.reduce_mean(abs(tf.subtract(s_ctx_v, s_predError_v)))

        s_loss_y = s_loss_pred_y + s_loss_ctx_y
        s_loss_u = s_loss_pred_u + s_loss_ctx_u
        s_loss_v = s_loss_pred_v + s_loss_ctx_v

        s_loss_yuv = s_loss_y + s_loss_u + s_loss_v

        # Texture Error
        t_pred_y = t_out_y[:, 0]
        t_pred_u = t_out_u[:, 0]
        t_pred_v = t_out_v[:, 0]
        t_ctx_y  = tf.nn.relu(t_out_y[:, 1])
        t_ctx_u  = tf.nn.relu(t_out_u[:, 1])
        t_ctx_v  = tf.nn.relu(t_out_v[:, 1])

        t_predError_y = abs(tf.subtract(t_pred_y, tf.squeeze(texture_y_gt, axis=1)))
        t_predError_u = abs(tf.subtract(t_pred_u, tf.squeeze(texture_u_gt, axis=1)))
        t_predError_v = abs(tf.subtract(t_pred_v, tf.squeeze(texture_v_gt, axis=1)))

        t_loss_pred_y = LAMBDA_Y * tf.reduce_mean(t_predError_y)
        t_loss_pred_u = LAMBDA_U * tf.reduce_mean(t_predError_u)
        t_loss_pred_v = LAMBDA_V * tf.reduce_mean(t_predError_v)

        t_loss_ctx_y = LAMBDA_Y * LAMBDA_CTX * tf.reduce_mean(abs(tf.subtract(t_ctx_y, t_predError_y)))
        t_loss_ctx_u = LAMBDA_U * LAMBDA_CTX * tf.reduce_mean(abs(tf.subtract(t_ctx_u, t_predError_u)))
        t_loss_ctx_v = LAMBDA_V * LAMBDA_CTX * tf.reduce_mean(abs(tf.subtract(t_ctx_v, t_predError_v)))

        t_loss_y = t_loss_pred_y + t_loss_ctx_y
        t_loss_u = t_loss_pred_u + t_loss_ctx_u
        t_loss_v = t_loss_pred_v + t_loss_ctx_v

        t_loss_yuv = t_loss_y + t_loss_u + t_loss_v

        # Optimizer
        t_vars = tf.trainable_variables()
        y_vars = [var for var in t_vars if '_y' in var.name]
        u_vars = [var for var in t_vars if '_u' in var.name]
        v_vars = [var for var in t_vars if '_v' in var.name]

        self.optimizer_y = tf.train.AdamOptimizer(LR).minimize(s_loss_y + t_loss_y, var_list=y_vars)
        self.optimizer_u = tf.train.AdamOptimizer(LR).minimize(s_loss_u + t_loss_u, var_list=u_vars)
        self.optimizer_v = tf.train.AdamOptimizer(LR).minimize(s_loss_v + t_loss_v, var_list=v_vars)
        self.optimizer_yuv = tf.train.AdamOptimizer(LR).minimize(s_loss_yuv + t_loss_yuv, var_list=t_vars)

        # Variables
        self.s_loss_y = s_loss_y
        self.s_loss_u = s_loss_u
        self.s_loss_v = s_loss_v
        self.s_loss_yuv = s_loss_yuv
        self.s_loss_pred_y = s_loss_pred_y
        self.s_loss_pred_u = s_loss_pred_u
        self.s_loss_pred_v = s_loss_pred_v
        self.s_loss_pred_yuv = s_loss_pred_v + s_loss_pred_u + s_loss_pred_v
        self.s_loss_ctx_y = s_loss_ctx_y
        self.s_loss_ctx_u = s_loss_ctx_u
        self.s_loss_ctx_v = s_loss_ctx_v
        self.s_loss_ctx_yuv = s_loss_ctx_y + s_loss_ctx_u + s_loss_ctx_v
        self.s_ctx_y = s_ctx_y
        self.s_ctx_u = s_ctx_u
        self.s_ctx_v = s_ctx_v

        self.t_loss_y = t_loss_y
        self.t_loss_u = t_loss_u
        self.t_loss_v = t_loss_v
        self.t_loss_yuv = t_loss_yuv
        self.t_loss_pred_y = t_loss_pred_y
        self.t_loss_pred_u = t_loss_pred_u
        self.t_loss_pred_v = t_loss_pred_v
        self.t_loss_pred_yuv = t_loss_pred_v + t_loss_pred_u + t_loss_pred_v
        self.t_loss_ctx_y = t_loss_ctx_y
        self.t_loss_ctx_u = t_loss_ctx_u
        self.t_loss_ctx_v = t_loss_ctx_v
        self.t_loss_ctx_yuv = t_loss_ctx_y + t_loss_ctx_u + t_loss_ctx_v
        self.t_ctx_y = t_ctx_y
        self.t_ctx_u = t_ctx_u
        self.t_ctx_v = t_ctx_v

    def crop_to_data(self, input_crop):

        # Parameters
        CTX_UP = self.args.ctx_up
        CTX_LEFT = self.args.ctx_left
        CTX_TOTAL = (CTX_LEFT * 2 + 1) * CTX_UP + CTX_LEFT
        BATCH_SIZE = self.args.batch_size
        CROP_SIZE = self.args.crop_size

        input_crop = tf.cast(input_crop, tf.float32)

        patch_size = int(CROP_SIZE * 2 / math.sqrt(BATCH_SIZE))

        # Obtain patch from cropped images
        random_patch = tf.extract_image_patches(input_crop, ksizes=[1, patch_size, patch_size, 1], strides=[1, patch_size, patch_size, 1], rates=[1,1,1,1], padding="VALID")
        random_patch = tf.reshape(random_patch, [-1, patch_size, patch_size, 3])
        data = tf.random_crop(random_patch, [BATCH_SIZE, CTX_UP+1, 2*CTX_LEFT+1, 3])
        data_reshape = tf.reshape(data, [BATCH_SIZE, -1, 3])

        # Obatin texture patch from patches
        support = data_reshape[:,:CTX_TOTAL,:]

        avg_support = tf.reduce_mean(support, axis=1, keepdims=True)
        avg_support = tf.tile(avg_support, [1,CTX_TOTAL,1])

        diff_support = tf.abs(support - avg_support)
        diff_support = tf.reduce_sum(diff_support, axis=(1,2))

        texture_threshold = tf.constant(50.0 * CTX_TOTAL)

        texture_indice = tf.where(tf.greater(diff_support, texture_threshold))
        smooth_indice = tf.where(tf.less(diff_support, texture_threshold))

        texture_crop = tf.gather_nd(data_reshape, texture_indice)

        # RGB2YUV
        data_rgb = data_reshape[:,:CTX_TOTAL+1,:]

        r,g,b = tf.split(data_rgb, 3, axis=2)

        u = b - tf.round((87 * r + 169 * g) / 256.0)
        v = r - g
        y = g + tf.round((86 * v + 29 * u) / 256.0)

        data_yuv = tf.concat([y,u,v], axis=2)
        data_yuv = tf.random.shuffle(data_yuv)

        # Subtract left pixel from support
        left_pixel = data_yuv[:,CTX_TOTAL-1,:]
        left_pixel = tf.expand_dims(left_pixel, axis=1)
        left_pixel = tf.tile(left_pixel, [1,CTX_TOTAL+1,1])

        data_yuv = data_yuv - left_pixel

        label = data_yuv[:,CTX_TOTAL,:]
        data_yuv = data_yuv[:,:CTX_TOTAL-1,:]

        data_y = data_yuv[:,:,0]
        data_u = data_yuv[:,:,1]
        data_v = data_yuv[:,:,2]

        input_data = tf.concat([data_y, data_u, data_v], axis=1)

        smooth_data = tf.gather_nd(input_data, smooth_indice)
        smooth_label = tf.gather_nd(label, smooth_indice)

        texture_data = tf.gather_nd(input_data, texture_indice)
        texture_label = tf.gather_nd(label, texture_indice)

        return smooth_data, smooth_label, texture_data, texture_label

    def train(self):

        GPU_NUM = self.args.gpu_num
        CKPT_DIR = self.args.ckpt_dir
        DATA_DIR = self.args.data_dir
        TENSORBOARD_DIR = self.args.tensorboard_dir
        LOAD = self.args.load
        CHANNEL_EPOCH = self.args.channel_epoch
        JOINT_EPOCH = self.args.joint_epoch
        BATCH_SIZE = self.args.batch_size
        PRINT_EVERY = self.args.print_every
        SAVE_EVERY = self.args.save_every

        # Assign GPU
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)

        global_step = tf.Variable(0, trainable=False)
        increase = tf.assign_add(global_step, 1)

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord, start=True)

            saver = tf.train.Saver(max_to_keep=1)
            ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
            writer = tf.summary.FileWriter(TENSORBOARD_DIR, graph=tf.get_default_graph())

            # Load model if trained before
            if ckpt and LOAD:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model Loaded")

            # Load dataset
            epoch = sess.run(global_step)

            s_loss_pred_epoch_y = s_loss_pred_epoch_u = s_loss_pred_epoch_v = 0
            s_loss_ctx_epoch_y = s_loss_ctx_epoch_u = s_loss_ctx_epoch_v = 0

            t_loss_pred_epoch_y = t_loss_pred_epoch_u = t_loss_pred_epoch_v = 0
            t_loss_ctx_epoch_y = t_loss_ctx_epoch_u = t_loss_ctx_epoch_v = 0

            while True:
                sess.run(increase)

                if epoch < CHANNEL_EPOCH :
                    if epoch == 0:
                        print("========== Train Y Channel ==========")
                    optimizer = self.optimizer_y
                elif epoch < 2*CHANNEL_EPOCH:
                    if epoch == CHANNEL_EPOCH:
                        print("========== Train U Channel ==========")
                    optimizer = self.optimizer_u
                elif epoch < 3*CHANNEL_EPOCH:
                    if epoch == 2*CHANNEL_EPOCH:
                        print("========== Train V Channel ==========")
                    optimizer = self.optimizer_v
                else:
                    if epoch == 3*CHANNEL_EPOCH:
                        print("========== Train YUV Channel ==========")
                    optimizer = self.optimizer_yuv

                _, s_loss_p_y, s_loss_p_u, s_loss_p_v, s_loss_c_y, s_loss_c_u, s_loss_c_v, t_loss_p_y, t_loss_p_u, t_loss_p_v, t_loss_c_y, t_loss_c_u, t_loss_c_v =\
                    sess.run([optimizer, self.s_loss_pred_y, self.s_loss_pred_u, self.s_loss_pred_v, self.s_loss_ctx_y, self.s_loss_ctx_u, self.s_loss_ctx_v,
                            self.t_loss_pred_y, self.t_loss_pred_u, self.t_loss_pred_v, self.t_loss_ctx_y, self.t_loss_ctx_u, self.t_loss_ctx_v])

                s_loss_pred_epoch_y += s_loss_p_y
                s_loss_pred_epoch_u += s_loss_p_u
                s_loss_pred_epoch_v += s_loss_p_v

                s_loss_ctx_epoch_y += s_loss_c_y
                s_loss_ctx_epoch_u += s_loss_c_u
                s_loss_ctx_epoch_v += s_loss_c_v

                t_loss_pred_epoch_y += t_loss_p_y
                t_loss_pred_epoch_u += t_loss_p_u
                t_loss_pred_epoch_v += t_loss_p_v

                t_loss_ctx_epoch_y += t_loss_c_y
                t_loss_ctx_epoch_u += t_loss_c_u
                t_loss_ctx_epoch_v += t_loss_c_v

                if (epoch + 1) % PRINT_EVERY == 0:
                    
                    s_loss_pred_epoch_y /= PRINT_EVERY
                    s_loss_pred_epoch_u /= PRINT_EVERY
                    s_loss_pred_epoch_v /= PRINT_EVERY
                    s_loss_ctx_epoch_y /= PRINT_EVERY
                    s_loss_ctx_epoch_u /= PRINT_EVERY
                    s_loss_ctx_epoch_v /= PRINT_EVERY
                    t_loss_pred_epoch_y /= PRINT_EVERY
                    t_loss_pred_epoch_u /= PRINT_EVERY
                    t_loss_pred_epoch_v /= PRINT_EVERY
                    t_loss_ctx_epoch_y /= PRINT_EVERY
                    t_loss_ctx_epoch_u /= PRINT_EVERY
                    t_loss_ctx_epoch_v /= PRINT_EVERY

                    s_loss_epoch_y = s_loss_pred_epoch_y + s_loss_ctx_epoch_y
                    s_loss_epoch_u = s_loss_pred_epoch_u + s_loss_ctx_epoch_u
                    s_loss_epoch_v = s_loss_pred_epoch_v + s_loss_ctx_epoch_v

                    t_loss_epoch_y = t_loss_pred_epoch_y + t_loss_ctx_epoch_y
                    t_loss_epoch_u = t_loss_pred_epoch_u + t_loss_ctx_epoch_u
                    t_loss_epoch_v = t_loss_pred_epoch_v + t_loss_ctx_epoch_v

                    print('%04d\n' % (epoch + 1),
                        '***Y Smooth***   lossPred=', '{:9.4f}'.format(s_loss_pred_epoch_y), 'lossContext=', '{:9.4f}'.format(s_loss_ctx_epoch_y), 'Loss=', '{:9.4f}'.format(s_loss_epoch_y), '***Y Texture***   lossPred=', '{:9.4f}'.format(t_loss_pred_epoch_y), 'lossContext=', '{:9.4f}'.format(t_loss_ctx_epoch_y), 'Loss=', '{:9.4f}\n'.format(t_loss_epoch_y),
                        '***U Smooth***   lossPred=', '{:9.4f}'.format(s_loss_pred_epoch_u), 'lossContext=', '{:9.4f}'.format(s_loss_ctx_epoch_u), 'Loss=', '{:9.4f}'.format(s_loss_epoch_u), '***U Texture***   lossPred=', '{:9.4f}'.format(t_loss_pred_epoch_u), 'lossContext=', '{:9.4f}'.format(t_loss_ctx_epoch_u), 'Loss=', '{:9.4f}\n'.format(t_loss_epoch_u),
                        '***V Smooth***   lossPred=', '{:9.4f}'.format(s_loss_pred_epoch_v), 'lossContext=', '{:9.4f}'.format(s_loss_ctx_epoch_v), 'Loss=', '{:9.4f}'.format(s_loss_epoch_v), '***V Texture***   lossPred=', '{:9.4f}'.format(t_loss_pred_epoch_v), 'lossContext=', '{:9.4f}'.format(t_loss_ctx_epoch_v), 'Loss=', '{:9.4f}\n'.format(t_loss_epoch_v),
                        '***YUV Smooth*** lossPred=', '{:9.4f}'.format(s_loss_pred_epoch_y + s_loss_pred_epoch_u + s_loss_pred_epoch_v), 'lossContext=',
                        '{:9.4f}'.format(s_loss_ctx_epoch_y + s_loss_ctx_epoch_u + s_loss_ctx_epoch_v), 'Loss=', '{:9.4f}'.format(s_loss_epoch_y + s_loss_epoch_u + s_loss_epoch_v),
                        '***YUV Texture*** lossPred=', '{:9.4f}'.format(t_loss_pred_epoch_y + t_loss_pred_epoch_u + t_loss_pred_epoch_v), 'lossContext=',
                        '{:9.4f}'.format(t_loss_ctx_epoch_y + t_loss_ctx_epoch_u + t_loss_ctx_epoch_v), 'Loss=', '{:9.4f}'.format(t_loss_epoch_y + t_loss_epoch_u + t_loss_epoch_v))

                    s_loss_pred_epoch_y = s_loss_pred_epoch_u = s_loss_pred_epoch_v = 0
                    s_loss_ctx_epoch_y = s_loss_ctx_epoch_u = s_loss_ctx_epoch_v = 0

                    t_loss_pred_epoch_y = t_loss_pred_epoch_u = t_loss_pred_epoch_v = 0
                    t_loss_ctx_epoch_y = t_loss_ctx_epoch_u = t_loss_ctx_epoch_v = 0

                if (epoch + 1) % SAVE_EVERY == 0:
                    saver.save(sess, CKPT_DIR + 'model_', global_step=epoch + 1)
                    self.print_weights('y')
                    self.print_weights('u')
                    self.print_weights('v')
                    print("Model Saved")

                epoch = sess.run(global_step)

            coord.request_stop()
            coord.join(threads)

    def print_weights(self, channel='y'):

        HIDDEN_UNIT = self.args.hidden_unit
        CTX_UP = self.args.ctx_up
        CTX_LEFT = self.args.ctx_left

        W_smooth = [v for v in tf.trainable_variables() if (('kernel' in v.name) and ('smooth_' + channel in v.name))]
        b_smooth = [v for v in tf.trainable_variables() if (('bias' in v.name) and ('smooth_' + channel in v.name))]

        W_texture = [v for v in tf.trainable_variables() if (('kernel' in v.name) and ('texture_' + channel in v.name))]
        b_texture = [v for v in tf.trainable_variables() if (('bias' in v.name) and ('texture_' + channel in v.name))]

        n_layer = len(W_smooth)

        W_s = []
        W_t = []
        b_s = []
        b_t = []

        for i in range(n_layer):
            W_s.append(W_smooth[i].eval())
            b_s.append(b_smooth[i].eval())
            W_t.append(W_texture[i].eval())
            b_t.append(b_texture[i].eval())

        n_in = W_s[0].shape[0]
        n_hidden = HIDDEN_UNIT
        n_out = W_smooth[-1].shape[1]

        filename = 'weights_smooth_' + channel + '.txt'

        f = open(filename, 'w')

        f.write(str(n_in) + '\n')
        f.write(str(n_hidden) + '\n')
        f.write(str(n_out) + '\n')
        f.write(str(n_layer) + '\n')
        f.write(str(CTX_UP) + '\n')
        f.write(str(CTX_LEFT) + '\n')

        for k in range(n_layer):
            for i in range(W_s[k].shape[0]):
                for j in range(W_s[k].shape[1]):
                    f.write(str(W_s[k][i,j]) + '\t')
                f.write('\n')

        for k in range(n_layer):
            for j in range(b_s[k].shape[0]):
                f.write(str(b_s[k][j]) + '\t')
            f.write('\n')


        f.close()

        filename = 'weights_texture_' + channel + '.txt'

        f = open(filename, 'w')

        f.write(str(n_in) + '\n')
        f.write(str(n_hidden) + '\n')
        f.write(str(n_out) + '\n')
        f.write(str(n_layer) + '\n')
        f.write(str(CTX_UP) + '\n')
        f.write(str(CTX_LEFT) + '\n')

        for k in range(n_layer):
            for i in range(W_t[k].shape[0]):
                for j in range(W_t[k].shape[1]):
                    f.write(str(W_t[k][i,j]) + '\t')
                f.write('\n')

        for k in range(n_layer):
            for j in range(b_t[k].shape[0]):
                f.write(str(b_t[k][j]) + '\t')
            f.write('\n')


        f.close()

    def print_all_weights(self):

        GPU_NUM = self.args.gpu_num
        CKPT_DIR = self.args.ckpt_dir

        # Assign GPU
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)
            ckpt = tf.train.get_checkpoint_state(CKPT_DIR)

            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model Loaded")

            self.print_weights('y')
            self.print_weights('u')
            self.print_weights('v')
                    

if __name__ == "__main__":

    args = parse_args()
    my_net = Network(args)
    my_net.build()
    my_net.train()
    my_net.print_all_weights()