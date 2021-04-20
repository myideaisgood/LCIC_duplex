import tensorflow as tf
import numpy as np
import cv2
import os
import tqdm

from config import parse_args

def read_dir(data_dir):

    dataset = []

    for dirpath, _, filenames in os.walk(data_dir):
        filenames = sorted(filenames)

        for img_filename in filenames:
            img_name = (data_dir + img_filename)
            
            dataset.append(img_name)
        
    return dataset

def _bytes_feature(value):
    "string / byte 타입을 받아서 byte list를 리턴"
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    "float / double 타입을 받아서 float list를 리턴"
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    "bool / enum / int / uint 타입을 받아서 int64 list를 리턴"
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_tfrecord(data_dir, image_list, tfrecords_name):

    print("Start converting image to tfrecords")
    writer = tf.python_io.TFRecordWriter(path=data_dir + tfrecords_name)

    # Data -> Feature -> Example -> Serialized Example -> Write

    for image_path in tqdm.tqdm(image_list):
        
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = np.array(image)
        height, width, _ = image.shape
        binary_image = image.tostring()

        feature_set = {
            'image': _bytes_feature(binary_image),
            'height': _int64_feature(height),
            'width': _int64_feature(width)
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature_set))

        serialized_example = example.SerializeToString()        

        writer.write(serialized_example)

    writer.close()

    print("End converting image to tfrecords")

def read_tfrecord(data_dir, tfrecords_name, num_epochs, batch_size, min_after_dequeue, crop_size=512):

    # Serialized Example -> Example -> Feature -> Data

    # filename queue
    filename_queue = tf.train.string_input_producer([data_dir + tfrecords_name], num_epochs=num_epochs)

    # read serialized examples
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # parse examples into features, each
    feature_set = {'image': tf.FixedLenFeature([], tf.string),
                      'height': tf.FixedLenFeature([], tf.int64),
                      'width': tf.FixedLenFeature([], tf.int64)}

    features = tf.parse_single_example(serialized_example, features=feature_set)

    # decode data
    img = tf.decode_raw(features['image'], tf.uint8)

    height = tf.cast(features['height'], tf.int64)
    width = tf.cast(features['width'], tf.int64)
    
    img = tf.reshape(img, [height, width, 3])
    img = tf.random_crop(img, [crop_size, crop_size, 3])

    # mini-batch examples queue
    img_input, height_input, width_input = tf.train.shuffle_batch([img, height, width], batch_size=batch_size, capacity=min_after_dequeue+3*batch_size, min_after_dequeue=min_after_dequeue)

    return img_input, height_input, width_input

def data_exist(data_dir, tfrecords_name):

    if os.path.exists(data_dir + tfrecords_name):
        return True
    else:
        return False

def show_samples(data_dir, image_list, tfrecords_name, epochs, batch_size, min_after_dequeue):

    if not data_exist(data_dir, tfrecords_name):
        write_tfrecord(data_dir, image_list, tfrecords_name)
    
    image, width, height = read_tfrecord(data_dir, tfrecords_name, num_epochs=epochs, batch_size=batch_size, min_after_dequeue=min_after_dequeue)

    with tf.Session() as sess:

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord, start=True)

        for i in range(epochs):
            for step in range(int(len(image_list) / batch_size)):
                img, w, h = sess.run([image, width, height])

                for j in range(batch_size):

                    img_each = img[j]
                    img_each = cv2.cvtColor(img_each, cv2.COLOR_RGB2BGR)

                    if j == 0:
                        img_show = img_each
                    else:
                        img_show = np.concatenate([img_show, img_each], axis=1)

                window_name = "Epoch_" + str(i) + "_Step_" + str(step)
                cv2.imshow(window_name, img_show)
                key = cv2.waitKey(0)

                if key == ord('q'):
                    coord.request_stop()
                    coord.join(threads)
                    return

                cv2.destroyAllWindows()

        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    
    args = parse_args()
    DATA_DIR = args.data_dir
    img_list = read_dir(args.data_dir + 'train/')

    tfrecords_name = 'train.tfrecord'

    epochs = 2
    batch_size = 4
    min_after_dequeue = 10

    show_samples(DATA_DIR, img_list, tfrecords_name, epochs, batch_size, min_after_dequeue)