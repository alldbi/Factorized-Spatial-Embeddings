from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import glob, os, random, math, collections, time, argparse, shutil
from utils.warp import feature_warping2, image_warping2
from matplotlib import cm


# Parameter setting ****************************************************************************************************

SAVE_FREQ = 500
SUMMARY_FREQ = 20
MODE = "test"
BATCH_SIZE = 32
DATA_DIRECTORY = '/media/lab320/0274E2F866ED37FC/dataset/CelebA/img_align_celeba'
LANDMARK_N = 32
DOWNSAMPLE_M = 4
DIVERSITY = 500.
ALIGN = 1.
LEARNING_RATE = 1.e-4
MOMENTUM = 0.5
RANDOM_SEED = 1234
WEIGHT_DECAY = 0.0005
SCALE_SIZE = 146
CROP_SIZE = 146
MAX_EPOCH = 200
OUTPUT_DIR = './OUTPUT'
CHECKPOINT = './backup/model/'



def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Factorized Spatial Embeddings")
    parser.add_argument("--mode", default=MODE, choices=["train", "test"])
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--input_dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the training or testing images.")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for adam.")
    parser.add_argument("--beta1", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--K", type=int, default=LANDMARK_N,
                        help="Number of landmarks.")
    parser.add_argument("--M", type=int, default=DOWNSAMPLE_M,
                        help="Downsampling value of the diversity loss.")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--diversity_weight", type=float, default=DIVERSITY,
                        help="Weight on diversity loss.")
    parser.add_argument("--align_weight", type=float, default=ALIGN,
                        help="Weight on align loss.")
    parser.add_argument("--scale_size", type=int, default=SCALE_SIZE,
                        help="Scale images to this size before cropping to CROP_SIZE")
    parser.add_argument("--crop_size", type=int, default=CROP_SIZE,
                        help="CROP images to this size")
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCH,
                        help="Number of training epochs")
    parser.add_argument("--checkpoint", default=CHECKPOINT,
                        help="Directory with checkpoint to resume training from or use for testing")
    parser.add_argument("--output_dir", default=OUTPUT_DIR,
                        help="Where to put output files")
    parser.add_argument("--summary_freq", type=int, default=SUMMARY_FREQ,
                        help="Update summaries every summary_freq steps")
    parser.add_argument("--save_freq", type=int, default=SAVE_FREQ, help="Save model every save_freq steps")
    return parser.parse_args()


def landmark_colors(n_landmarks):
    """Compute landmark colors.

    Returns:
      An array of RGB values.
    """
    cmap = cm.get_cmap('hsv')
    landmark_color = []
    landmark_color.append((0., 0., 0.))
    for i in range(n_landmarks):
        landmark_color.append(cmap(i/float(n_landmarks))[0:3])
    landmark_color = np.array(landmark_color)
    return landmark_color


# Collections definition
Examples = collections.namedtuple("Examples",
                                  "paths, images, images_deformed, deformation, count, steps_per_epoch, shape")
Model = collections.namedtuple("Model", "pos_loss, neg_loss, distance")

def weight_decay():
    """Compute weight decay loss.

    Returns:
      Weight decay loss.
    """
    costs = []
    for var in tf.trainable_variables():
        if var.op.name.find('filter')>0:
            costs.append(tf.nn.l2_loss(var))
    return tf.add_n(costs)

def conv(batch_input, out_channels, stride=1):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [5, 5, in_channels, out_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        conv = tf.nn.conv2d(batch_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv


def save_images(fetches, args, step=None):
    image_dir = os.path.join(args.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        filename = name + "-" + "outputs" + ".png"
        if step is not None:
            filename = "%08d-%s" % (step, filename)
        fileset["outputs"] = filename
        out_path = os.path.join(image_dir, filename)
        contents = fetches["outputs"][i]
        with open(out_path, "wb") as f:
            f.write(contents)
        filesets.append(fileset)
    return filesets


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1

def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

def load_examples(args):
    """Load all images in the input_dir.

    Returns:
      Examples.paths : batch of path of images,
      Examples.images : batch of images,
      Examples.images_deformed : batch of deformed images,
      Examples.deformation : batch of deformation parameters,
    """
    if args.input_dir is None or not os.path.exists(args.input_dir):
        raise Exception("input_dir does not exist")

    decode = tf.image.decode_jpeg
    # load distorted pairs address
    input_paths = glob.glob(os.path.join(args.input_dir, "*.jpg"))

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle= args.mode == "train")
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)
        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have required channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        images = preprocess(raw_input)

    seed = random.randint(0, 2 ** 31 - 1)

    # scale and crop input image to match 256x256 size
    def transform(image):
        r = image
        r = tf.image.resize_images(r, [args.scale_size, args.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, args.scale_size - args.crop_size + 1, seed=seed)), dtype=tf.int32)
        if args.scale_size > args.crop_size:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], args.crop_size, args.crop_size)

        elif args.scale_size < args.crop_size:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("images"):
        input_images = transform(images)
        if args.mode=="train":
            input_images, _ = image_warping2(input_images, w=0.0)
        deformed_images, deformation = image_warping2(input_images, w=0.1)
        deformation = tf.squeeze(deformation)

        # crop after warping
        input_images = tf.image.crop_to_bounding_box(input_images, 5, 5, 128, 128)
        deformed_images = tf.image.crop_to_bounding_box(deformed_images, 5, 5, 128, 128)

        # clip image values
        input_images = tf.clip_by_value(input_images, clip_value_min=-1., clip_value_max=1.)
        deformed_images = tf.clip_by_value(deformed_images, clip_value_min=-1., clip_value_max=1.)

    paths_batch, images_batch, images_deformed_batch, deformation_batch = tf.train.batch(
        [paths, input_images, deformed_images, deformation], batch_size=args.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / args.batch_size))

    return Examples(
        paths=paths_batch,
        images=images_batch,
        images_deformed=images_deformed_batch,
        deformation=deformation_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
        shape=raw_input.get_shape()
    )

def CNN_tower(inputs, n_landmarks, isTrain):

    n_filters = [20, 48, 64, 80, 256, n_landmarks]
    with tf.variable_scope("layer_1"):
        x = conv(inputs, n_filters[0])
        x = tf.contrib.layers.batch_norm(x, updates_collections=None, decay=0.9, center=True,
                                                     scale=True,
                                                     activation_fn=tf.nn.relu, is_training=isTrain)
        # only the first layer has a 2x2 maxpooling
        x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)
    with tf.variable_scope("layer_2"):
        x = conv(x, n_filters[1])
        x = tf.contrib.layers.batch_norm(x, updates_collections=None, decay=0.9, center=True,
                                         scale=True,
                                         activation_fn=tf.nn.relu, is_training=isTrain)
    with tf.variable_scope("layer_3"):
        x = conv(x, n_filters[2])
        x = tf.contrib.layers.batch_norm(x, updates_collections=None, decay=0.9, center=True,
                                         scale=True,
                                         activation_fn=tf.nn.relu, is_training=isTrain)
    with tf.variable_scope("layer_4"):
        x = conv(x, n_filters[3])
        x = tf.contrib.layers.batch_norm(x, updates_collections=None, decay=0.9, center=True,
                                         scale=True,
                                          activation_fn=tf.nn.relu, is_training=isTrain)
    with tf.variable_scope("layer_5"):
        x = conv(x, n_filters[4])
        x = tf.contrib.layers.batch_norm(x, updates_collections=None, decay=0.9, center=True,
                                         scale=True,
                                         activation_fn=tf.nn.relu, is_training=isTrain)
    with tf.variable_scope("layer_6"):
        x = conv(x, n_filters[5])
        x = tf.contrib.layers.batch_norm(x, updates_collections=None, decay=0.9, center=True,
                                         scale=True,
                                         activation_fn=tf.nn.relu, is_training=isTrain)

    return x


def align_loss(predA_deformed, predB, n_landmarks):


    # compute the mean of landmark locations


    batch_size = predB.get_shape()[0]
    pred_size = predB.get_shape()[1]
    index = tf.range(0, tf.cast(pred_size, tf.float32), delta=1, dtype=tf.float32)
    index = tf.reshape(index, [pred_size, 1])

    x_index = tf.tile(index, [1, pred_size])

    index = tf.transpose(index)

    y_index = tf.tile(index, [pred_size, 1])

    x_index = tf.expand_dims(x_index, 2)
    x_index = tf.expand_dims(x_index, 0)

    y_index = tf.expand_dims(y_index, 2)
    y_index = tf.expand_dims(y_index, 0)

    x_index = tf.tile(x_index, [batch_size, 1, 1, n_landmarks])
    y_index = tf.tile(y_index, [batch_size, 1, 1, n_landmarks])


    x_index_avg_A = x_index * predA_deformed
    y_index_avg_A = y_index * predA_deformed

    x_index_avg_B = x_index * predB
    y_index_avg_B = y_index * predB


    pA_sum = tf.reduce_sum(predA_deformed, axis=[1, 2])
    pB_sum = tf.reduce_sum(predB, axis=[1, 2])


    x_index_avg_A = tf.reduce_mean(x_index_avg_A, axis=[1, 2])
    y_index_avg_A = tf.reduce_mean(y_index_avg_A, axis=[1, 2])
    x_index_avg_B = tf.reduce_mean(x_index_avg_B, axis=[1, 2])
    y_index_avg_B = tf.reduce_mean(y_index_avg_B, axis=[1, 2])

    x_index_avg_A = x_index_avg_A / pA_sum
    y_index_avg_A = y_index_avg_A / pA_sum
    x_index_avg_B = x_index_avg_B / pB_sum
    y_index_avg_B = y_index_avg_B / pB_sum

    # compute align loss
    loss = tf.pow(x_index_avg_A-x_index_avg_B, 2.) + tf.pow(y_index_avg_A - y_index_avg_B, 2.)
    loss = tf.reduce_mean(loss)
    return loss, x_index, y_index


def align_loss2(predA, predB, deformation, n_landmarks):


    # compute the mean of landmark locations

    batch_size = predA.get_shape()[0]
    pred_size = predA.get_shape()[1]
    index = tf.range(0, tf.cast(pred_size, tf.float32), delta=1, dtype=tf.float32)
    index = tf.reshape(index, [pred_size, 1])

    x_index = tf.tile(index, [1, pred_size])

    index = tf.transpose(index)

    y_index = tf.tile(index, [pred_size, 1])

    x_index = tf.expand_dims(x_index, 2)
    x_index = tf.expand_dims(x_index, 0)

    y_index = tf.expand_dims(y_index, 2)
    y_index = tf.expand_dims(y_index, 0)

    x_index = tf.tile(x_index, [batch_size, 1, 1, n_landmarks])
    y_index = tf.tile(y_index, [batch_size, 1, 1, n_landmarks])


    u_norm2 = tf.pow(x_index, 2.) + tf.pow(y_index, 2.)
    u_norm2 = u_norm2 * predA
    loss_part1 = tf.reduce_sum(u_norm2, axis=[1, 2])

    x_index_deformed = feature_warping2(x_index, deformation, padding=3)
    y_index_defomred = feature_warping2(y_index, deformation, padding=3)
    v_norm2 = tf.pow(x_index_deformed, 2.) + tf.pow(y_index_defomred, 2.)
    v_norm2 = v_norm2 * predB
    loss_part2 = tf.reduce_sum(v_norm2, axis=[1, 2])


    loss_part3x = tf.reduce_sum(x_index * predA, axis=[1, 2])
    loss_part3y = tf.reduce_sum(y_index * predA, axis=[1, 2])
    loss_part4x = tf.reduce_sum(x_index_deformed * predB, axis=[1, 2])
    loss_part4y = tf.reduce_sum(y_index_defomred * predB, axis=[1, 2])

    loss_part3 = loss_part3x * loss_part4x + loss_part3y * loss_part4y
    loss = loss_part1 + loss_part2 - 2. * loss_part3
    loss = tf.reduce_mean(loss)

    return loss




def main():

    """Create the model and start the training."""
    args = get_arguments()


    tf.set_random_seed(args.random_seed)
    examples = load_examples(args)

    print("examples count = %d" % examples.count)



    with tf.variable_scope("cnn_tower"):
        predA = CNN_tower(examples.images, n_landmarks=args.K, isTrain=args.mode == "train")

    with tf.variable_scope("cnn_tower", reuse=True):
        predB = CNN_tower(examples.images_deformed, n_landmarks=args.K, isTrain=args.mode == "train")


    # apply a spatial softmax to obtain K probability maps

    pred_size = predA.get_shape()[1]

    predA = tf.reshape(predA, [-1, pred_size*pred_size, args.K])
    predB = tf.reshape(predB, [-1, pred_size*pred_size, args.K])

    predA = tf.nn.softmax(predA, axis=1)
    predB = tf.nn.softmax(predB, axis=1)

    predA = tf.reshape(predA, [-1, pred_size, pred_size, args.K])
    predB = tf.reshape(predB, [-1, pred_size, pred_size, args.K])


    # visualizing landmarks
    predA_vis = tf.reduce_mean(predA, axis=3)
    predA_vis = tf.expand_dims(predA_vis, axis=3)

    # another visualization
    pred_max = tf.reduce_max(predA, axis=[1, 2])
    pred_max = tf.expand_dims(pred_max, axis=1)
    pred_max = tf.expand_dims(pred_max, axis=1)
    pred_max = tf.equal(predA, pred_max)
    pred_max = tf.cast(pred_max, tf.float32)

    mask = tf.range(start=1, limit=args.K+1, delta=1, dtype=tf.float32)
    mask = tf.reshape(mask, [1, 1, 1, args.K])
    mask = tf.tile(mask, [args.batch_size, pred_size, pred_size, 1])
    mask = mask * pred_max
    mask = tf.reduce_max(mask, axis=3, keepdims=True)

    landmarks = tf.convert_to_tensor(landmark_colors(args.K), tf.float32)

    mask = tf.reshape(mask, [args.batch_size, pred_size*pred_size])
    mask = tf.cast(mask, tf.int32)
    mask = tf.gather(landmarks, mask, axis=0)
    mask = tf.reshape(mask, [args.batch_size, pred_size, pred_size, 3])

    pred_max = tf.reduce_max(pred_max, axis=3)
    pred_max = tf.expand_dims(pred_max, axis=3)

    # compute the diversity loss


    def diversity_loss(pred, n_landmark, pool_size):
        pred_pool = tf.nn.pool(pred, window_shape=[pool_size, pool_size], strides=[1, 1], pooling_type="AVG", padding="VALID")
        # convert avg pool to sum pool
        # pred_pool = pred_pool * float(pool_size) * float(pool_size)
        pred_max = tf.reduce_max(pred_pool, axis=3)
        pred_max_sum = tf.reduce_sum(pred_max, axis=[1, 2])
        pred_max_sum = float(n_landmark) - pred_max_sum
        pred_max_sum = tf.reduce_mean(pred_max_sum)
        return pred_max_sum

    diversityLoss_predA = diversity_loss(predA, n_landmark=args.K, pool_size=args.M)
    diversityLoss_predB = diversity_loss(predB, n_landmark=args.K, pool_size=args.M)
    div_loss = diversityLoss_predA + diversityLoss_predB

    # compute the align loss
    algn_loss = align_loss2(predA, predB, examples.deformation, n_landmarks= args.K)

    # compute the weight decay loss
    decay_loss = weight_decay() * args.weight_decay


    with tf.name_scope("train"):
        optim = tf.train.AdamOptimizer(args.learning_rate, args.beta1)
        # grads_and_vars = optim.compute_gradients(loss)
        # train = optim.apply_gradients(grads_and_vars)
        train_op = optim.minimize(algn_loss*args.align_weight + div_loss*args.diversity_weight + decay_loss )
    # global_step = tf.contrib.framework.get_or_create_global_step()
    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)
    train = tf.group(train_op, incr_global_step)

    input_images = deprocess(examples.images)
    input_deformed = deprocess(examples.images_deformed)


    # overlay landmarks on the input image

    landmarks_image = pred_max * mask

    pred_max_resized = tf.image.resize_images(pred_max, [128, 128], tf.image.ResizeMethod.AREA)
    pred_max_resized = tf.greater(pred_max_resized, 0.)
    pred_max_resized = tf.cast(pred_max_resized, tf.float32)

    mask_resized = tf.image.resize_images(mask, [128, 128])


    input_images_landmark = input_images * (1.-pred_max_resized) + pred_max_resized * mask_resized


    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    tf.summary.image("Input", input_images)
    tf.summary.image("Deformed", input_deformed)
    tf.summary.image("PredA", predA_vis)
    # tf.summary.image("AApredAmax", mask)
    # tf.summary.image("PredB", predB_vis)
    tf.summary.image("Landmark", input_images_landmark)
    # tf.summary.image("AApredAmax", landmarks_image)

    tf.summary.scalar("loss_align", algn_loss)
    tf.summary.scalar("loss_diversity", div_loss)
    tf.summary.scalar("loss_decay", decay_loss)

    output_images = tf.image.convert_image_dtype(input_images_landmark, dtype=tf.uint8, saturate=True)
    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "outputs": tf.map_fn(tf.image.encode_png, output_images, dtype=tf.string, name="input_pngs"),
        }


    saver = tf.train.Saver(max_to_keep=1)

    sv = tf.train.Supervisor(logdir=os.path.join(os.path.join(args.output_dir, 'logs')), save_summaries_secs=0, saver=None)

    with sv.managed_session() as sess:

        max_steps = 2 ** 32
        if args.max_epochs is not None:
            max_steps = examples.steps_per_epoch * args.max_epochs
            print ("max epochs: ", args.max_epochs)
            print ("max steps : ", max_steps)
            start = time.time()

        print("parameter_count =", sess.run(parameter_count))

        if args.checkpoint is not None:
            print ("loading from checkpoint...")
            checkpoint = tf.train.latest_checkpoint(args.checkpoint)
            saver.restore(sess, checkpoint)

        if args.mode == "train":
            # training
            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                fetches = {
                    "train": train,
                    "global_step": sv.global_step,
                    "loss": algn_loss,
                    "labels": examples.images,
                    "offset": examples.deformation,
                    "predA" : predA,
                    "decay_loss":decay_loss,
                    "div_loss":div_loss,


                }

                if should(freq=args.summary_freq):
                    fetches["summary"] = sv.summary_op

                results = sess.run(fetches)

                if should(freq=args.summary_freq):
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * args.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * args.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (
                    train_epoch, train_step, rate, remaining / 60))
                    print ("loss_align", results["loss"])
                    print ("loss_diversity", results["div_loss"])
                    print ("loss_decay", results["decay_loss"])
                    print ("------------------------------")

                if should(freq=args.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(args.output_dir, "model"), global_step=sv.global_step)
        elif args.mode=="test":
            # testing
            start = time.time()
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results, args)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
            print("rate", (time.time() - start) / max_steps)


if __name__ == '__main__':
    main()
