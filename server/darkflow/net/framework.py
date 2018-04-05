import cv2
import numpy as np
from . import yolo
from ..utils.box import BoundBox
from ..cython_utils.cy_yolo2_findboxes import box_constructor
from os.path import basename

class framework(object):
    
    def __init__(self, meta, FLAGS):
        model = basename(meta['model'])
        model = '.'.join(model.split('.')[:-1])
        meta['name'] = model
        
        self.meta = meta
        self.FLAGS = FLAGS

    def is_inp(self, file_name):
        return True

class YOLOv2(object):

    def __init__(self, meta, FLAGS):
        model = basename(meta['model'])
        model = '.'.join(model.split('.')[:-1])
        meta['name'] = model
        
        self.constructor(meta, FLAGS)

    def findboxes(self, net_out):
	    # meta
        meta = self.meta
        boxes = list()
        boxes=box_constructor(meta,net_out)
        return boxes


    def resize_input(self, im):
        h, w, c = self.meta['inp_size']
        imsz = cv2.resize(im, (w, h))
        imsz = imsz / 255.
        imsz = imsz[:, :, ::-1]
        return imsz


    def process_box(self, b, h, w, threshold):
        max_indx = np.argmax(b.probs)
        max_prob = b.probs[max_indx]
        label = self.meta['labels'][max_indx]
        if max_prob > threshold:
            left = int((b.x - b.w / 2.) * w)
            right = int((b.x + b.w / 2.) * w)
            top = int((b.y - b.h / 2.) * h)
            bot = int((b.y + b.h / 2.) * h)
            if left < 0:  left = 0
            if right > w - 1: right = w - 1
            if top < 0:   top = 0
            if bot > h - 1:   bot = h - 1
            mess = '{}'.format(label)
            return (left, right, top, bot, mess, max_indx, max_prob)
        return None


    def preprocess(self, im, allobj=None):
        """
	    Takes an image, return it as a numpy tensor that is readily
	    to be fed into tfnet. If there is an accompanied annotation (allobj),
	    meaning this preprocessing is serving the train process, then this
	    image will be transformed with random noise to augment training data,
	    using scale, translation, flipping and recolor. The accompanied
	    parsed annotation (allobj) will also be modified accordingly.
	    """
        if type(im) is not np.ndarray:
            im = cv2.imread(im)

        if allobj is not None:  # in training mode
            result = imcv2_affine_trans(im)
            im, dims, trans_param = result
            scale, offs, flip = trans_param
            for obj in allobj:
                _fix(obj, dims, scale, offs)
                if not flip: continue
                obj_1_ = obj[1]
                obj[1] = dims[0] - obj[3]
                obj[3] = dims[0] - obj_1_
            im = imcv2_recolor(im)

        im = self.resize_input(im)
        if allobj is None: return im
        return im  # , np.array(im) # for unit testing


    def constructor(self, meta, FLAGS):

        def _to_color(indx, base):
            """ return (b, r, g) tuple"""
            base2 = base * base
            b = 2 - indx / base2
            r = 2 - (indx % base2) / base
            g = 2 - (indx % base2) % base
            return (b * 127, r * 127, g * 127)
        if 'labels' not in meta:
            misc.labels(meta, FLAGS) #We're not loading from a .pb so we do need to load the labels
        assert len(meta['labels']) == meta['classes'], (
		    'labels.txt and {} indicate' + ' '
		    'inconsistent class numbers'
	    ).format(meta['model'])

	    # assign a color for each label
        colors = list()
        base = int(np.ceil(pow(meta['classes'], 1./3)))
        for x in range(len(meta['labels'])): 
            colors += [_to_color(x, base)]
        meta['colors'] = colors
        self.fetch = list()
        self.meta, self.FLAGS = meta, FLAGS

	    # over-ride the threshold in meta if FLAGS has it.
        if FLAGS.threshold > 0.0:
            self.meta['thresh'] = FLAGS.threshold