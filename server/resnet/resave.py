import tensorflow as tf
import imagenet_input as data_input
import resnet

imgs = tf.constant(-1.0, shape=[1, 1, 1024, 1024, 1])
#img = tf.reshape(img, [1980, 1980, 1])
#_img = data_input.resize_image_and_batch(img)
#_img = tf.image.convert_image_dtype(_img, tf.float32)

#input = list()
#results = list()
#for b in boxes:
#	boxResults = self.process_box(b, h, w, threshold)
#	if boxResults is not None:
#		results.append(boxResults)
#		left, right, top, bot, mess, max_indx, confidence = boxResults
#		cropped = img[left:right, top:bot, :]
#		reshaped = data_input.resize_image_and_batch(cropped)
#		input.append(reshaped)
#imgs = tf.concat(input, axis=1)
	#img = sess.run(cropped)
	#fig = plt.figure()
	#fig.add_subplot(1,2,1)
	#plt.imshow(im)
	#fig.add_subplot(1,2,2)
	#plt.imshow(img)
	#plt.show()

hp = resnet.HParams(batch_size=32,
            num_gpus=1,
            num_classes=66000,
            weight_decay=0.00001,
            momentum=0.9,
            finetune=False)
global_step = tf.Variable(0, trainable=False, name='global_step')
network_test = resnet.ResNet(hp, imgs, None, global_step, name="test", reuse_weights=False, training=False)
network_test.build_model()

# Build an initialization operation to run below.
init = tf.global_variables_initializer()

# Start running operations on the Graph.
sess = tf.Session(config=tf.ConfigProto(
					gpu_options = tf.GPUOptions(
					per_process_gpu_memory_fraction=0.99,
					allow_growth=True),
				allow_soft_placement=False,
            	# allow_soft_placement=True,
            	log_device_placement=False))
sess.run(init)

# Create a saver.
saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000)
print('Load checkpoint %s' % './train/0_1000/model.ckpt-374000')
saver.restore(sess, './train/0_1000/model.ckpt-374000')
init_step = global_step.eval(session=sess)
#preds = sess.run([network_test.preds])

vars = tf.contrib.framework.list_variables('./train/0_1000/')
with tf.Graph().as_default(), tf.Session().as_default() as sess:

  new_vars = []
  for name, shape in vars:
    v = tf.contrib.framework.load_variable('./train/0_1000/', name)
    new_vars.append(tf.Variable(v, name='g1/'+name))
    

  saver = tf.train.Saver(new_vars)
  sess.run(tf.global_variables_initializer())
  saver.save(sess, './resave/model')