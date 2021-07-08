import numpy as np
import tensorflow as tf
import logging

import i3d

logging.basicConfig(level = logging.INFO)

def evaluate_input(input_path):

    _IMAGE_SIZE = 224

    _SAMPLE_VIDEO_FRAMES = 79

    _CHECKPOINT_PATHS = {
            'rgb600': '..../checkpoint_kinetics600/model.ckpt',
    }

    _LABEL_MAP_PATH_600 = '....../Kinetics600 classes.txt'

    #a = b = []
    FLAGS = tf.flags.FLAGS

    flags_dict = FLAGS._flags()    
    val_list = [val for val in flags_dict]    
    for val in val_list:
        FLAGS.__delattr__(val)

    tf.reset_default_graph()

    tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, rgb600, flow, or joint')
    tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')

    eval_type = FLAGS.eval_type

    imagenet_pretrained = FLAGS.imagenet_pretrained

    NUM_CLASSES = 600
    
    if eval_type not in ['rgb', 'rgb600', 'flow', 'joint']:
        raise ValueError('Bad `eval_type`, must be one of rgb, rgb600, flow, joint')

    kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH_600)]
    
    if eval_type in ['rgb', 'rgb600', 'joint']:
        # RGB input has 3 channels.
        rgb_input = tf.placeholder(
                tf.float32,
                shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))      
        
        with tf.variable_scope('RGB'):
            
            rgb_model = i3d.InceptionI3d(
                NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
            
            rgb_logits, _ = rgb_model(
                rgb_input, is_training=False, dropout_keep_prob=1.0)
            
        
        rgb_variable_map = {}
        for variable in tf.global_variables():

            if variable.name.split('/')[0] == 'RGB':
                rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable

        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    model_predictions = tf.nn.softmax(rgb_logits)

    
    with tf.Session() as sess:
        feed_dict = {}
        
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb600'])
        logging.info('RGB checkpoint restored')
        rgb_sample = np.load(input_path)
        logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))
        feed_dict[rgb_input] = rgb_sample

        out_logits, out_predictions = sess.run(
            [rgb_logits, model_predictions],
            feed_dict=feed_dict)

        out_logits = out_logits[0]
        out_predictions = out_predictions[0]
        sorted_indices = np.argsort(out_predictions)[::-1]

        print('Norm of logits: %f' % np.linalg.norm(out_logits))
        for index in sorted_indices[:5]:
            print('Score:{:.4f} Logits:{:.4f} Class:{}'.format(out_predictions[index], out_logits[index], kinetics_classes[index]))

    return (sorted_indices, out_predictions)
    
def main(input_path):
    evaluate_input(input_path)  

if __name__ == '__main__':
    #input_path = '/data//v_CricketShot_g04_c01_rgb.npy'
    evaluate_input()  
else:
    input_path = '/data/v_CricketShot_g04_c01_rgb.npy'
    main(input_path)
