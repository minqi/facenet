# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import align.detect_face
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

def save_model(model, inputs, outputs, export_path):
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    signature_inputs = {}
    signature_outputs = {}
    for k, v in inputs.iteritems():
        tensor_info = tf.saved_model.utils.build_tensor_info(model.get_layer(v))
        signature_inputs[k] = tensor_info

    for k, v in outputs.iteritems():
        tensor_info = tf.saved_model.utils.build_tensor_info(model.get_layer(v))
        signature_outputs[k] = tensor_info

    prediction_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
          inputs=signature_inputs,
          outputs=signature_outputs,
          method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
          tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              prediction_signature 
      },
      )
    builder.save()

def load_net(sess, inputs, outputs, export_path):
    signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    meta_graph_def = tf.saved_model.loader.load(
               sess,
              [tf.saved_model.tag_constants.SERVING],
              export_path)
    signature = meta_graph_def.signature_def

    run_inputs = []
    for k in inputs:
        tensor_name = signature[signature_key].inputs[k].name
        tensor = sess.graph.get_tensor_by_name(tensor_name)
        run_inputs.append(tensor)

    run_outputs = []
    for k in outputs:
        tensor_name = signature[signature_key].outputs[k].name
        tensor = sess.graph.get_tensor_by_name(tensor_name)
        run_outputs.append(tensor)

    return lambda img : sess.run(run_outputs, dict(zip(run_inputs, [img])))

with tf.Graph().as_default():
  
    sess = tf.Session()
    with sess.as_default():
        with tf.variable_scope('pnet'):
            data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
            pnet = align.detect_face.PNet({'data':data})
            pnet.load('src/align/det1.npy', sess)
        with tf.variable_scope('rnet'):
            data = tf.placeholder(tf.float32, (None,24,24,3), 'input')
            rnet = align.detect_face.RNet({'data':data})
            rnet.load('src/align/det2.npy', sess)
        with tf.variable_scope('onet'):
            data = tf.placeholder(tf.float32, (None,48,48,3), 'input')
            onet = align.detect_face.ONet({'data':data})
            onet.load('src/align/det3.npy', sess)

        # import pdb; pdb.set_trace()
            
        # pnet_fun = lambda img : sess.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'), feed_dict={'pnet/input:0':img})
        # rnet_fun = lambda img : sess.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'), feed_dict={'rnet/input:0':img})
        # onet_fun = lambda img : sess.run(('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0'), feed_dict={'onet/input:0':img})

        pnet_fun = load_net(sess, ['img_in'], ['conv_out', 'prob_out'], './pnet_saved_model')
        rnet_fun = load_net(sess, ['img_in'], ['conv_out', 'prob_out'], './rnet_saved_model')
        onet_fun = load_net(sess, ['img_in'], ['conv1_out', 'conv2_out', 'prob_out'], './onet_saved_model')

    # save model
    # save_model(pnet, {'img_in': 'data'}, {'conv_out': 'conv4-2', 'prob_out': 'prob1'}, './pnet_saved_model')
    # save_model(rnet, {'img_in': 'data'}, {'conv_out': 'conv5-2', 'prob_out': 'prob1'}, './rnet_saved_model')
    # save_model(onet, {'img_in': 'data'}, {'conv1_out': 'conv6-2', 'conv2_out': 'conv6-3', 'prob_out': 'prob1'}, './onet_saved_model')

minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

source_path = '/Users/minqijiang/Desktop/test2.jpg'
img = misc.imread(source_path)

bounding_boxes, points = align.detect_face.detect_face(img, minsize, pnet_fun, rnet_fun, onet_fun, threshold, factor)

print('Bounding box: %s' % bounding_boxes)
print('Points: %s' % points)

n = len(points)
x = [points[i] for i in xrange(0, n//2, 1)]
y = [points[i] for i in xrange(n//2, n, 1)]


print(x)
print(y)

im = plt.imread(source_path)
fig,ax = plt.subplots(1)
ax.imshow(im,zorder=1)
ax.scatter(x,y,zorder=2, s=3)

p = patches.Rectangle(
    (bounding_boxes[0][0], bounding_boxes[0][1]), 
    bounding_boxes[0][2] - bounding_boxes[0][0], 
    bounding_boxes[0][3] - bounding_boxes[0][1], 
    edgecolor='r',facecolor='none'
)
pc = PatchCollection([p], 
        zorder=3,
        edgecolor='green',facecolor='none')
ax.add_collection(pc)

plt.show()


