import tensorflow as tf
import logging
from 3D CNN import eval3dcnn
from 2D CNN import eval2dcnn

logging.basicConfig(level = logging.INFO)
#from 2D CNN.eval_sample-RGB import *

pre_processed_path = '...../v_CricketShot_g04_c01_rgb.npy'
video_path = '...../v_TennisSwing_g14_c03.avi'
label_path = '......./Kinetics600 classes.txt'
label_map = [x.strip() for x in open(label_path)]

def main():

	indices, scores= eval3dcnn.evaluate_input(pre_processed_path)

	top1_score_3dcnn = scores[indices[0]]
	top1_class = label_map[indices[0]]

	top1_labels_2dcnn, top1_score_2dcnn = eval2dcnn.run(video_path)

	for i in range(len(top1_score_2dcnn)):

		total_score += top1_score_2dcnn[i]

	avg_2d_score = total_score/len(top1_score_2dcnn)

	avg_fusion_score = 1.5*(top1_score_3dcnn) + 1.0*(avg_2d_score) / 2.5

	print('Average softmax fusion score {} and class {}'.format(avg_fusion_score, top1_class))


if __name__ == '__main__':
	main()
