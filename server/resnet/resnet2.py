import tensorflow as tf
import resnet.resnet as resnet


class ResNet18(object):
    def __init__(self):
        graph = tf.Graph()
        with graph.as_default() as g:
            with g.name_scope('g1') as scope:

			    # Build an initialization operation to run below.
                init = tf.global_variables_initializer()

			    # Start running operations on the Graph.
                self.sess = tf.Session(config=tf.ConfigProto(
								    gpu_options = tf.GPUOptions(
								    per_process_gpu_memory_fraction=0.99,
								    allow_growth=True),
							    allow_soft_placement=False,
            				    # allow_soft_placement=True,
            				    log_device_placement=False))
                self.sess.run(init)

                # Build network
                self.hp = resnet.HParams(batch_size=32,
                            num_gpus=1,
                            num_classes=88712,
                            weight_decay=0.00001,
                            momentum=0.9,
                            finetune=False)
			    #global_step = tf.Variable(0, trainable=False, name='global_step')
                self.input = tf.placeholder(tf.float32, shape=[1, None, 64, 224, 1])
                self.network = resnet.ResNet(self.hp, self.input, None, None, name="test", reuse_weights=False, training=False)
                self.network.build_model()

			    # Create a saver.
                checkpoint = './resnet/train/model.ckpt-466000'
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000)
                print('Load checkpoint %s' % checkpoint)
                saver.restore(self.sess, checkpoint)
			    #init_step = global_step.eval(session=sess)