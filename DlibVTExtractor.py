import numpy as np
import pandas as pd
import dlib
import cv2
import os
from scipy import interpolate

class DlibVTExtractor(object):
	def __init__(self):
		pass

	def get_equi_dist(self, contour, M):
		contour = np.array(contour)
		d = np.diff(contour, axis=0)
		dist_from_vertex_to_vertex = np.hypot(d[:, 0], d[:, 1])
		cumulative_dist_along_path = np.concatenate(([0], np.cumsum(dist_from_vertex_to_vertex, axis=0)))
		dist_steps = np.linspace(0, cumulative_dist_along_path[-1], M)
		contour = interpolate.interp1d(cumulative_dist_along_path, contour, axis=0)(dist_steps)

		return contour

	def prepare_train_data(self, src_dir, dst_dir):
		if not os.path.exists(dst_dir):
			os.mkdir(dst_dir)
		
		f1_files = [f for f in os.listdir(src_dir) if (f.endswith('.avi') and 'f1' in f)]
		f5_files = [f for f in os.listdir(src_dir) if (f.endswith('.avi') and 'f5' in f)]
		m3_files = [f for f in os.listdir(src_dir) if (f.endswith('.avi') and 'm3' in f)]
		avi_files = np.concatenate((f1_files, f5_files, m3_files))

		with open(os.path.join(dst_dir, 'vt_shape_dlib.xml'), 'w') as out_xml:
			out_xml.write("<?xml version='1.0' encoding='ISO-8859-1'?>\n")
			out_xml.write("<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>\n") 
			out_xml.write("<dataset>\n") 
			out_xml.write("<name> Training VT </name>\n") 
			out_xml.write("<images>\n") 
			for file in avi_files:
				vid_name = os.path.join(src_dir, file)
				cap = cv2.VideoCapture(vid_name)
				fnum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

				gtname = os.path.join(src_dir, file[:-4] + '.csv')
				gt = pd.read_csv(gtname, header=None)
				gt = gt.iloc[:, :].values
				pnts = gt[:fnum, :160]

				lands = gt[:fnum, 161:]

				for f in range(fnum):
					ret, frame = cap.read()
					cv2.imwrite(os.path.join(dst_dir, file[:-4] + '_frame_' + str(f) + '.jpg'), frame)

					vt_cont = np.reshape(pnts[f, :], (80, 2), order='F')
					vt_cont = self.get_equi_dist(vt_cont, 80)
					vt_cont = np.array(vt_cont, dtype=np.int32)

					vt_land = np.reshape(lands[f, :], (6, 2), order='F')
					vt_land = np.array(vt_land, np.int32)

					x, y, w, h = cv2.boundingRect(vt_cont)

					out_xml.write("<image file='{}'>\n".format(file[:-4] + '_frame_' + str(f) + '.jpg'))
					out_xml.write("<box top='{}' left='{}' width='{}' height='{}'>\n".format(x - 5, y - 5, w + 10, h + 10))

					for j in range(80):
						out_xml.write("<part name='{}' x='{}' y='{}'/>\n".format(str(j), vt_cont[j, 0], vt_cont[j, 1]))

					out_xml.write("</box>\n")
					out_xml.write("</image>\n")
			out_xml.write("</images>\n")
			out_xml.write("</dataset>\n")

	def train_shape_predictor(self, output_name, input_xml):

		print('[INFO] trining shape predictor....')
		# get the training options
		options = dlib.shape_predictor_training_options()
		options.tree_depth = 4
		options.nu = 0.1
		options.cascade_depth = 15
		options.feature_pool_size = 400
		options.num_test_splits = 100
		options.oversampling_amount = 10
		options.be_verbose = True 
		options.num_threads = 4 

		# start training the model
		dlib.train_shape_predictor(input_xml, output_name, options)

	def train_shape_detector(self, output_name, input_xml):

		print('[INFO] training shape detector....')
		# get the training options
		options = dlib.simple_object_detector_training_options()
		options.add_left_right_image_flips = False
		options.C = 5
		options.num_threads = 4
		options.be_verbose = True

		# start training the model
		dlib.train_simple_object_detector(input_xml, output_name, options)



if __name__=="__main__":
	estimator = DlibVTExtractor()
	estimator.prepare_train_data('data/usc-timit', 'data/dlib')
	estimator.train_shape_detector('models/dlib/dlib_vt_detector.svm', 'data/dlib/vt_shape_dlib.xml')
	estimator.train_shape_predictor('models/dlib/dlib_vt_predictor.dat', 'data/dlib/vt_shape_dlib.xml')
	