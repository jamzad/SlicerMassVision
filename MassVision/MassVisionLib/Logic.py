# from cProfile import label
# from lib2to3.refactor import get_fixers_from_package
from math import pi
import os
import SimpleITK as sitk
# from pyexpat import model
# import unittest
import logging
import vtk, qt, ctk, slicer
from vtk.util import numpy_support

try:
	import matplotlib
except ModuleNotFoundError:
	slicer.util.pip_install("matplotlib")
	import matplotlib

import matplotlib.pyplot as plt
import matplotlib.cm as cm
## fix Mac crash
matplotlib.use('Agg')

# try:
# 		import cv2
# except ModuleNotFoundError:
# 		slicer.util.pip_install("opencv-python")
# 		import cv2
try:
	from PIL import Image as PILImage
except:
	slicer.util.pip_install("pillow")
	from PIL import Image as PILImage

try:
	from sklearn.decomposition import PCA
except ModuleNotFoundError:
	slicer.util.pip_install("scikit-learn")
	from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.cross_decomposition import PLSRegression, PLSCanonical
from sklearn.utils import resample
from sklearn.cluster import KMeans

try:
	import pandas as pd
except ModuleNotFoundError:
	slicer.util.pip_install("pandas")
	import pandas as pd

try:
	from tqdm import tqdm
except ModuleNotFoundError:
	slicer.util.pip_install("tqdm")
	from tqdm import tqdm
		
import numpy as np

from slicer.ScriptedLoadableModule import *
from scipy.special import softmax, expit
from scipy.signal import find_peaks
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle

try:
	import h5py
except ModuleNotFoundError:
	slicer.util.pip_install("h5py")
	import h5py

from scipy.stats import ttest_ind

from MassVisionLib.Utils import *


class MassVisionLogic(ScriptedLoadableModuleLogic):
	"""This class should implement all the actual
	computation done by your module.  The interface
	should be such that other python code can import
	this class and make use of the functionality without
	requiring an instance of the Widget.
	Uses ScriptedLoadableModuleLogic base class, available at:
	https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
	"""
 # stores the data we want 
	def __init__(self):
		"""
		Called when the logic class is instantiated. Can be used for initializing member variables.
		"""
		ScriptedLoadableModuleLogic.__init__(self)

		# all of the variables required for this class
		self.peaks = None
		self.peaks_norm = None
		self.dim_x = None
		self.dim_y = None
		self.path = None
		self.mz = None
		self.ionpath = None
		self.slideName = None
		self.savenameBase = None
		self.saveFolder = None
		self.df= None
		self.iondims = None
		# self.peaks_3D = None
		self.modellingFile = None
		self.split = 'random'
		self.test_cases = set()
		self.train_cases = set()
		self.val_cases = set()
		# self.selectedmz = []
		self.model_type = None
		self.train_balancing = 'None'
		#self.volume = None
		self.CNNHyperparameters = {}
		self.REIMS_H = 300
		self.lastPCA = None
		self.contrast_thumbnail_inds = None
		self.pixel_clusters = None
		self.peaks_pca = None
		

	def setDefaultParameters(self, parameterNode):
		"""
		Initialize parameter node with default settings.
		"""
		if not parameterNode.GetParameter("Threshold"):
			parameterNode.SetParameter("Threshold", "100.0")
		if not parameterNode.GetParameter("Invert"):
			parameterNode.SetParameter("Invert", "false")

	def process(self, inputVolume, outputVolume, imageThreshold, invert=False, showResult=True):
		"""
		Run the processing algorithm.
		Can be used without GUI widget.
		:param inputVolume: volume to be thresholded
		:param outputVolume: thresholding result
		:param imageThreshold: values above/below this threshold will be set to 0
		:param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
		:param showResult: show output volume in slice viewers
		"""

		if not inputVolume or not outputVolume:
			raise ValueError("Input or output volume is invalid")

		import time
		startTime = time.time()
		logging.info('Processing started')

		# Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
		cliParams = {
			'InputVolume': inputVolume.GetID(),
			'OutputVolume': outputVolume.GetID(),
			'ThresholdValue' : imageThreshold,
			'ThresholdType' : 'Above' if invert else 'Below'
			}
		cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
		# We don't need the CLI module node anymore, remove it to not clutter the scene with it
		slicer.mrmlScene.RemoveNode(cliNode)

		stopTime = time.time()
		logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')

	def MSI_h52numpy(self, h5_file):
		with h5py.File(h5_file, 'r') as h5file:
			peaks = h5file['peaks'][:]
			mz = h5file['mz'][:]

		dim_y, dim_x, _ = peaks.shape
		peaks = peaks.reshape((dim_y*dim_x,-1),order='C')

		return peaks, mz, dim_y, dim_x

	def MSI_csv2numpy(self, csv_file):
		df = pd.read_csv(csv_file)
		peak_start_col = 2
		mz = np.array(df.columns[peak_start_col:], dtype='float')
		peaks = df[df.columns[peak_start_col:]].values
		loc =  df[df.columns[0:peak_start_col]].values
		dim_y = int(float(df.columns[0].split('=')[-1]))
		dim_x = int(float(df.columns[1].split('=')[-1]))

		# handle unsorted and missing values
		flat_ind = list(loc[:,0]*dim_x+loc[:,1])
		full_len = dim_y*dim_x

		ind_pad = set(range(full_len))-set(flat_ind)

		flat_ind.extend(list(ind_pad))
		peaks = np.pad(peaks, ((0,len(ind_pad)),(0,0)), 'empty')
		peaks = peaks[np.argsort(flat_ind)]

		return peaks, mz, dim_y, dim_x

	# takes in the desi text function and organized the information
	# into the peaks, mz values, xdimensions, ydimensions
	def DESI_txt2numpy(self, desi_text):
		data = []
		with open(desi_text, 'r') as read_obj:
			for i,line in enumerate(read_obj):
				x = line.split()
				y = [float(num) for num in x]
				data.append(y)
						
		ind = np.argsort(data[3]) # data[3] has unsorted m/z values
		mz = np.take_along_axis(np.asarray(data[3]), ind, axis=0) # sort with indices

		x, y = [], []
		peaks = []
		
		for i in range(4,len(data)-1):
			x.append(data[i][1])
			y.append(data[i][2])
			p = np.asarray(data[i][3:-2])
			p = np.take_along_axis(p, ind, axis=0)
			p = np.expand_dims(p,axis=0)
			peaks.append(p)
		peaks = np.concatenate(peaks,axis=0)

		## find desi data dimension
		t = np.asarray(x)
		t = np.abs(np.diff(t))
		dim_x = int(np.round(np.max(t)/np.min(t)))+1
		t = np.asarray(y)
		dim_y = int(np.round(np.abs(t[0]-t[-1])/np.max(np.abs(np.diff(t)))))+1
		
		return peaks, mz, dim_y, dim_x
			
	# noramlizes the peaks 
	def tic_normalize(self, peaks):
		tot_ion_cur = np.sum(peaks, axis=1)
		peaks_ticn = np.empty(peaks.shape)
		for i in range(len(peaks)):
			if tot_ion_cur[i]!=0:
				peaks_ticn[i] = peaks[i]/tot_ion_cur[i]
		return peaks_ticn

	# nomalization of spectrum to a specific ion (spectrum based)
	def ref_normalize(self,peaks, mz, mz_ref):
		"""
		nomalization of spectrum to a specific ion (spectrum based)
		author: @moon
		"""
	
		peak_ref = peaks[:,mz == mz_ref]
		peaks_norm = np.zeros(peaks.shape)
		for i in range(len(peaks)):
			if peak_ref[i]!=0:
				peaks_norm[i] = peaks[i]/(0.1+peak_ref[i])
		return peaks_norm
 
	def single_slide_pixel_aggregation(self, peaks, labels, roi_width, roi_stride, agg_func, partial_threshold):
		"""
		aggregation of pixel-level spectra for a single slide
		author: @moon
		"""
	
		labels_roi = []
		labels_center = []
		peaks_agg = []
	
		## find the center of rois in an even grid
		x_min, x_max = np.min(labels[:,2]), np.max(labels[:,2])
		y_min, y_max = np.min(labels[:,3]), np.max(labels[:,3])
		x_centers = np.arange(x_min+roi_width//2, x_max-roi_width//2, roi_stride)
		y_centers = np.arange(y_min+roi_width//2, y_max-roi_width//2, roi_stride)
	
		for x_cent in tqdm(x_centers, leave=False):
			for y_cent in y_centers:
				ind_center = (labels[:,2]==x_cent) & (labels[:,3]==y_cent)
				if np.any(ind_center):
					ind_roi = (labels[:,2]>=x_cent-roi_width//2) & \
								(labels[:,2]<x_cent-roi_width//2+roi_width) & \
								(labels[:,3]>=y_cent-roi_width//2) & \
								(labels[:,3]<y_cent-roi_width//2+roi_width)
	
					if (np.sum(ind_roi) >= partial_threshold/100*roi_width*roi_width) and (len(np.unique(labels[ind_roi,1]))==1):
						labels_center.append(labels[ind_center])
						labels_roi.append(labels[ind_roi])
						temp = peaks[ind_roi]
						# temp = agg_func(temp, axis=0, keepdims=True)
						# peaks_agg.append(self.tic_normalize(temp))
						peaks_agg.append(agg_func(temp, axis=0, keepdims=True))
	
		labels_center = np.concatenate(labels_center)
		labels_roi = np.concatenate(labels_roi)
		peaks_agg = np.concatenate(peaks_agg)
	
		sort_ind = np.lexsort((labels_center[:,3],labels_center[:,2], labels_center[:,1]))
		labels_center = labels_center[sort_ind]
		peaks_agg = peaks_agg[sort_ind]
	
		return peaks_agg, labels_center, labels_roi

	# convert/save segemntations to color-coded images
	def labels_to_masks(self,labels_single, all_classes, max_range, image_save_info=None):
		"""
		convert/save segemntations to color-coded images
		author: @moon
		"""
	
		x_single = labels_single[:,2]
		y_single = labels_single[:,3]
		roi_reconstruct = np.full(max_range, 'none', dtype=object)
		for i in range(len(x_single)):
			roi_reconstruct[y_single[i], x_single[i]] = labels_single[i,1]
	
		class2num = LabelEncoder().fit(all_classes)
		class_single = class2num.transform(labels_single[:,1])
		class_single = class_single+1
		roi_reconstruct_num = np.zeros(max_range)
		for i in range(len(x_single)):
			roi_reconstruct_num[y_single[i], x_single[i]] = class_single[i]
	
		if image_save_info!=None:
			
			n_colors = len(class2num.classes_)
			cmap_colors = cm.get_cmap('jet')(np.linspace(0, 1, n_colors))
			cmap_custom = cm.colors.ListedColormap( np.vstack(([0,0,0,1], cmap_colors)) )
			plt.figure(figsize=[6,6])
			plt.imshow(roi_reconstruct_num.T, cmap=cmap_custom, 
					   interpolation=None, vmin=-0.5, vmax=len(class2num.classes_)+0.5)
			plt.axis('off')
			cbar = plt.colorbar(ticks=np.arange(len(class2num.classes_))+1)
			cbar.ax.set_yticklabels(class2num.classes_)


			save_folder, save_suffix = image_save_info


			save_name = os.path.join(save_folder,str(labels_single[0,0])+save_suffix)
			plt.savefig(save_name, bbox_inches='tight', dpi=300)
			plt.close()

		return roi_reconstruct_num, roi_reconstruct

	# the whole postporocessing fuction including nomalization, band filtering, and pixel aggregation
	def dataset_post_processing(self, spec_normalization, subband_selection, pixel_aggregation, processed_dataset_name):
		"""
		the whole postporocessing fuction including nomalization, band filtering, and pixel aggregation
		author: @moon
		"""
		
		# load csv dataset
		df = self.csv_processing

		# extract information
		peak_start_col = 4
		mz = np.array(df.columns[peak_start_col:], dtype='float')
		peaks = df[df.columns[peak_start_col:]].values
		labels =  df[df.columns[0:peak_start_col]].values 

		# handle missing values
		peaks = np.nan_to_num(peaks)

		# spectrum nrmalization
		print("spec_normalization",spec_normalization)
		if spec_normalization != None:
			if spec_normalization == 'tic':
				peaks = self.tic_normalize(peaks)
			else:
				peaks = self.ref_normalize(peaks=peaks, mz=mz, mz_ref=float(spec_normalization))
			print('mass spectra normalization done!')

		# spectrum range filtering
		if subband_selection != None:
			lower_band, upper_band = subband_selection
			ind_subband = (mz>=lower_band) & (mz<=upper_band)
			mz = mz[ind_subband]
			peaks = peaks[:,ind_subband]
			print('m/z range filtering done!')

		# pixel aggregation
		if pixel_aggregation != None:
			print('pixel aggregation ...')
			
			roi_width, roi_stride, agg_mode, partial_threshold = pixel_aggregation

			agg_func = getattr(np, agg_mode)

			slide_names = np.unique(labels[:,0])

			save_folder = processed_dataset_name[:-4]+'_AGGMAPS'
			if not os.path.exists(save_folder):
				os.makedirs(save_folder)

			peaks_agg, labels_agg = [], []
			for slide_name in tqdm(slide_names):
				ind_single_slide = labels[:,0] == slide_name
				peaks_single_slide = peaks[ind_single_slide]
				labels_single_slide = labels[ind_single_slide]

				peaks_agg_single_slide, labels_center_single_slide, labels_roi_single_slide =  self.single_slide_pixel_aggregation(peaks_single_slide, labels_single_slide, 
																				roi_width, roi_stride, agg_func, partial_threshold)

				peaks_agg.append(peaks_agg_single_slide)
				labels_agg.append(labels_center_single_slide)
				
				all_classes = np.unique(labels[:,1])
				max_range = [labels_single_slide[:,3].max()+1, labels_single_slide[:,2].max()+1]
				_,_ = self.labels_to_masks(labels_single_slide, all_classes, max_range, image_save_info=[save_folder, '_original'])
				_,_ = self.labels_to_masks(labels_center_single_slide, all_classes, max_range, image_save_info=[save_folder, '_center'])
				_,_ = self.labels_to_masks(labels_roi_single_slide, all_classes, max_range, image_save_info=[save_folder, '_roi'])
		
			peaks = np.concatenate(peaks_agg)
			labels = np.concatenate(labels_agg)
			print('pixel aggregation done!')

		# save the processed csv dataset
		csv_data = np.concatenate( [labels, peaks] , axis=1)
		csv_column = list(df.columns[:peak_start_col])+list(mz)
		df = pd.DataFrame(csv_data, columns=csv_column)
		print('saving the processed csv ...')
		df.to_csv(processed_dataset_name, index=False)
		print('processed csv successfully saved')

		retstr = self.datasetInfo(df)

		return retstr

	def model_deployment(self, spec_normalization, pixel_aggregation, dep_mask):

		# align peaks to the model referecne m/z list
		if set(self.DmzRef) == set(self.mz):
			aligned_peaks, mz_map = self.peaks, self.mz
		else:
			aligned_peaks, mz_map = self.peak_alignemnt_to_reference(self.mz, self.peaks, self.DmzRef, thresh=0.05)
		
		# take care of missing values
		aligned_peaks = np.nan_to_num(aligned_peaks)

		## pre-processing
		# spectrum nrmalization
		if spec_normalization != None:
			if spec_normalization == 'tic':
				aligned_peaks = self.tic_normalize(aligned_peaks)
			else:
				aligned_peaks = self.ref_normalize(peaks=aligned_peaks, mz=self.DmzRef, mz_ref=float(spec_normalization))
			print('mass spectra normalization done!')

		# pixel aggregation
		if pixel_aggregation != None:

			agg_width = pixel_aggregation[0]
			p_before = agg_width//2
			agg_func = getattr(np, pixel_aggregation[1])

			aligned_peaks_3d = aligned_peaks.reshape((self.dim_y,self.dim_x,-1),order='C')

			aligned_peaks_3d_agg = np.zeros_like(aligned_peaks_3d)
			for y_ind in tqdm(range(self.dim_y)):
				for x_ind in range(self.dim_x):
					y_min = max(0, y_ind-p_before)
					y_max = min(self.dim_y, y_ind-p_before+agg_width)
					x_min = max(0, x_ind-p_before)
					x_max = min(self.dim_x, x_ind-p_before+agg_width)
					pixel_agg = aligned_peaks_3d[y_min:y_max, x_min:x_max, :]
					aligned_peaks_3d_agg[y_ind, x_ind, :] = agg_func(pixel_agg, axis=(0,1), keepdims=True)

			aligned_peaks = aligned_peaks_3d_agg.reshape((self.dim_y*self.dim_x,-1),order='C')
			del aligned_peaks_3d_agg, aligned_peaks_3d

		## ion normalizer
		ion_normalizer = self.Dpipeline[0]
		aligned_peaks_norm = ion_normalizer.transform(aligned_peaks)
		aligned_peaks_norm = np.clip(aligned_peaks_norm, 0, 1) #if minmax

		## model pipeline
		if self.Dmodel_type=='PCA-LDA':
			aligned_peaks_pca = self.Dpipeline[1].transform(aligned_peaks_norm)
			aligned_peaks_preds = self.Dpipeline[2].predict(aligned_peaks_pca)
		elif self.Dmodel_type in ['Random Forest', 'SVM']:
			aligned_peaks_preds = self.Dpipeline[1].predict(aligned_peaks_norm)
		elif self.Dmodel_type=='PLS-DA':
			aligned_peaks_prob = self.Dpipeline[1].predict(aligned_peaks_norm)
			aligned_peaks_preds = self.Dpipeline[2].inverse_transform(aligned_peaks_prob)
			aligned_peaks_preds = aligned_peaks_preds.ravel()

		## encode class labels to numbers
		label_to_num = LabelEncoder().fit(self.Dclass_order)
		aligned_peaks_preds_num = label_to_num.transform(aligned_peaks_preds)
		preds_img = aligned_peaks_preds_num.reshape((self.dim_y,self.dim_x),order='C')

		## visualize predictions
		n_colors = len(self.Dclass_order)
		if n_colors<=10:
			class_colors = plt.cm.tab10(range(n_colors))
		else:
			class_colors = cm.get_cmap('jet_r')(np.linspace(0, 1, n_colors))
		cmap_custom = cm.colors.ListedColormap( class_colors )


		# masked deployment
		if dep_mask != None:
			segmentationNode = slicer.util.getNodesByClass('vtkMRMLSegmentationNode')[0]
			segmentation = segmentationNode.GetSegmentation()
			segIDs = segmentation.GetSegmentIDs()
			segNames = [segmentation.GetSegment(segID).GetName() for segID in segIDs]

			segID = segIDs[segNames==dep_mask]
			mask_array = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segID)
			mask_array = mask_array[0]

			preds_img = np.where(mask_array==1, preds_img, np.nan)
			cmap_custom.set_bad(color='black',alpha=0)

		fig, ax = plt.subplots()
		cax = ax.imshow(preds_img, cmap=cmap_custom, vmin=-0.5, vmax=n_colors-0.5) 
		ax.axis('off')
		divider = make_axes_locatable(ax)
		cbar_ax = divider.append_axes("right", size="5%", pad=0.1)
		cbar = fig.colorbar(cax, cax=cbar_ax)
		cbar.set_ticks(np.arange(n_colors))
		cbar.set_ticklabels(self.Dclass_order)

		plt.savefig(self.savenameBase + '_deployment.png', bbox_inches='tight', dpi=600)
		plt.close()
		
		if self.Dmodel_type=='PCA-LDA':
			lda_image = self.Dpipeline[2].transform(aligned_peaks_pca)
			lda_image = MinMaxScaler().fit_transform(lda_image)
			lda_image = lda_image.reshape((self.dim_y,self.dim_x,-1),order='C')
			if lda_image.shape[2]==2:
				lda_image = np.pad(lda_image, [(0,0), (0,0), (0,1)])
			elif lda_image.shape[2]>=3:
				lda_image = lda_image[:,:,:3]
			
			# if dep_mask != None:
			# 	lda_image = lda_image * mask_array

			lda_image = np.expand_dims(lda_image, axis=0)*255
			self.visualizationRunHelper(lda_image, lda_image.shape, visualization_type='deploy_lda')

			# # display deployment results
			# RedCompNode = slicer.util.getNode("vtkMRMLSliceCompositeNodeRed")
			# RedNode = slicer.util.getNode("vtkMRMLSliceNodeRed")

			# ldaImageNode = slicer.util.loadVolume(self.savenameBase + '_deploy_lda.jpeg', {"singleFile": True})
			# slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutSideBySideView)

			# RedCompNode.SetBackgroundVolumeID(ldaImageNode.GetID())
			# RedNode.SetOrientation("Axial")
			# YellowCompNode.SetBackgroundVolumeID(volumeNode.GetID())
			# YellowNode.SetOrientation("Axial")
			# slicer.util.resetSliceViews()

    
		# display deployment results
		YellowCompNode = slicer.util.getNode("vtkMRMLSliceCompositeNodeYellow")
		YellowNode = slicer.util.getNode("vtkMRMLSliceNodeYellow")

		volumeNode = slicer.util.loadVolume(self.savenameBase + '_deployment.png', {"singleFile": True})
		slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpYellowSliceView)

		YellowCompNode.SetBackgroundVolumeID(volumeNode.GetID())
		YellowNode.SetOrientation("Axial")
		slicer.util.resetSliceViews()


	def peak_alignemnt_to_reference(self, mz_test, peaks_test, mz_ref, thresh=0.05):

		"""""
		function:
			MSI m/z alighnemnt
			align m/z values and peaks of an MSI data to a reference m/z list
		inputs:
			mz_test - m/z list of the slide
			peaks_test - peak array of slide
			mz_ref - reference m/z list
			thresh - maximum m/z distance for inclusion
		output:
			aligned_peaks - aligned peak array to the reference m/z list
			final_mz - shows how the slide m/z list is aligned to the reference list 
		author: @moon
		"""""

		# calculate peak abundancy
		pmean = peaks_test.mean(axis=0)/peaks_test.mean(axis=0).max()

		# generate list for multiple m/z
		n_mz = len(mz_ref)
		new_mz = []
		for j in range(n_mz):
			new_mz.append([])

		# generate equivalent abundancy list
		new_peaks = []
		for j in range(n_mz):
			new_peaks.append([])

		print('aligning m/z to the reference ...')
		# align current m/z to the reference m/z
		for j in range(len(mz_test)):
			x = mz_test[j]
			y = pmean[j]
			diff = np.abs(mz_ref-x)
			if diff.min()<thresh:
				ind = diff.argmin()
				new_mz[ind].append(x)
				new_peaks[ind].append(y)

		# convert to pandas dataframe for simpler handling
		new_mz_df = pd.DataFrame([new_mz], columns=mz_ref)
		new_peaks_df = pd.DataFrame([new_peaks], columns=mz_ref)

		# eliminate the multiple m/z based on aboundance
		final_mz = np.nan*np.ones(n_mz,)
		for j in range(n_mz):
			mz_cell = new_mz_df[mz_ref[j]][0]
			if len(mz_cell)==0:
				pass
			elif len(mz_cell)==1:
				final_mz[j] = mz_cell[0]
			else:
				pmean_cell = new_peaks_df[mz_ref[j]][0]
				i_abundant = np.array(pmean_cell).argmax()
				final_mz[j] = mz_cell[i_abundant]

		print('aligning the peaks ...')
		# align peaks accordingly
		aligned_peaks = np.nan*np.ones([len(peaks_test),n_mz])
		for j,mz_val in enumerate(tqdm(final_mz)):
			if ~np.isnan(mz_val):
				p_ind = np.where(mz_test == mz_val)[0]
				aligned_peaks[:,j] = peaks_test[:,p_ind].flat

		print('alignment to the m/z reference done!')
		
		return aligned_peaks, final_mz


	## if used as .T it will normalize based on spectra
	#.t = transpose if transpose and do this instead of ion will use the spectral
	# if 2d array can decide which way i want sum and std to be 1
	def ion_minmax_normalize(self, peaks):
			max_ion_int = np.max(peaks, axis=0)
			min_ion_int = np.min(peaks, axis=0)
			peaks_ionorm = np.empty(peaks.shape)
			for i in range(peaks.shape[1]):
					if max_ion_int[i]!=min_ion_int[i]:
							peaks_ionorm[:,i] = (peaks[:,i]-min_ion_int[i])/(max_ion_int[i]-min_ion_int[i])
			return peaks_ionorm

	def ion_zscore_normalize(self, peaks):
			scaler = StandardScaler()
			peaks_feat_norm = scaler.fit_transform(peaks)
			return peaks_feat_norm

	def pca_plot_mask(self,peaks_pca,mask_ind,dim_y,dim_x):
			peaks_pca = self.ion_minmax_normalize(peaks_pca)
			pca_img = np.zeros([dim_y*dim_x,3])
			for i in range(len(mask_ind)):
					pca_img[mask_ind[i],:] = peaks_pca[i,:]
			pca_image = pca_img.reshape((dim_y,dim_x,-1),order='C')
			
			return pca_image
	
	# normalizes the peaks
	def normalize(self):
			self.peaks_norm = self.ion_minmax_normalize(self.tic_normalize(self.peaks))
			return True
		
	# generates and displays the pca image
	def pca_display(self):
		# generates and displays the pca image
		dim_reduction = PCA(n_components=3)
		peaks_pca = dim_reduction.fit_transform(self.peaks_norm)
		peaks_pca = MinMaxScaler().fit_transform( peaks_pca )
		self.peaks_pca = peaks_pca
		pca_image = peaks_pca.reshape((self.dim_y,self.dim_x,-1),order='C')
		pca_image = np.expand_dims(pca_image, axis=0)*255
		
		self.visualizationRunHelper(pca_image, pca_image.shape, visualization_type='pca')

		self.lastPCA = dim_reduction
		return True
	
	def LoadingsRank(self):
		pca = self.lastPCA
		mz = self.mz
		n_ranks = 5
		colors = [' (Red)', ' (Green)', ' (Blue)']
		info = ''
		loadings = pca.components_
		contrast_thumbnail_inds = []
		for pc_ind in range(len(loadings)):
			# print('PC'+str(pc_ind+1))
			info += 'PC'+str(pc_ind+1)+colors[pc_ind]+'\n'
			feature_contributions = np.abs(loadings)
			most_important_ind = np.argsort(-feature_contributions[pc_ind]) 
			# print('all:', mz[most_important_ind][:n_ranks])
			info += 'all: '+', '.join([str(x) for x in mz[most_important_ind][:n_ranks]])+'\n'
			# contrast_thumbnail_inds.append(most_important_ind[:n_ranks])

			pos_feature_contributions = np.maximum(loadings[pc_ind],0)
			pos_important_ind = np.argsort(-pos_feature_contributions)
			# print('pos:', mz[pos_important_ind][:n_ranks])
			info += 'pos: '+', '.join([str(x) for x in mz[pos_important_ind][:n_ranks]])+'\n'
			contrast_thumbnail_inds.append(pos_important_ind[:n_ranks])

			neg_feature_contributions = np.minimum(loadings[pc_ind],0)
			neg_important_ind = np.argsort(neg_feature_contributions)
			# print('neg:', mz[neg_important_ind][:n_ranks])
			info += 'neg: '+', '.join([str(x) for x in mz[neg_important_ind][:n_ranks]])+'\n\n'
			contrast_thumbnail_inds.append(neg_important_ind[:n_ranks])

		self.contrast_thumbnail_inds = np.concatenate(contrast_thumbnail_inds)
		return info
	
	# generates the single ion image for the m/z value specified
	def single_ion_display_colours(self, mz_r):
		# generates and displays the single ion image
		# ch_r = self.selectedmz.index(mz_r)
		ch_r = list(self.mz).index(mz_r)
		image_r = (self.peaks_norm[:,ch_r]).reshape((self.dim_y,self.dim_x,-1),order='C')

		# displays the single ion image
		voxelType = vtk.VTK_UNSIGNED_CHAR
		imageData = vtk.vtkImageData()

		# gets the dimensions the correct way
		# try instead of doing this mackenzie just get the shape and flip the dimension of image
		reversed = tuple(list(image_r.shape)[::-1])
		imageData.SetDimensions(reversed)
		self.iondims = reversed

		imageData.AllocateScalars(voxelType, 1)
		imageData.GetPointData().GetScalars().Fill(0)

		return image_r
	
	def singleIonVisualization(self, mz, heatmap):
		#mz_ind = self.selectedmz.index(mz)
		#slicer.modules.markups.logic().JumpSlicesToLocation(self.volume[mz_ind], True)
		array = self.single_ion_display_colours(mz)
		array = np.transpose(array, (2, 0, 1))
		self.visualizationRunHelper(array, array.shape, 'single', heatmap=heatmap)
		return True
	
	def ViewAbundanceThumbnail(self):
		total_abundance = self.tic_normalize(self.peaks).sum(axis=0)
		sorted_indices = np.argsort(-total_abundance)

		dim_y = self.dim_y
		dim_x = self.dim_x
		n_row = 7
		n_col = 8
		fig_scale = 2

		fig, axes = plt.subplots(n_row, n_col, figsize=(fig_scale*n_col/dim_y*dim_x, fig_scale*n_row), gridspec_kw={'wspace': 0, 'hspace': 0})

		for i, ax in enumerate(axes.flat):
			if i==0:
				tic_image = self.peaks.sum(axis=1).reshape((dim_y,dim_x),order='C')
				tic_image = tic_image[::2,::2]
				ax.imshow(tic_image, cmap='gray')
				ax.text(0, 0, 'TIC', color='yellow', fontsize=10, ha='left', va='top', 
						bbox=dict(facecolor='black', alpha=0.9, boxstyle='round,pad=0.3'))  # Add label
			else:
				ion_image = self.peaks_norm[:, sorted_indices[i-1]].reshape((dim_y,dim_x),order='C')
				ion_image = ion_image[::2,::2]
				ax.imshow(ion_image, cmap='inferno')
				ax.text(0, 0, str(self.mz[ sorted_indices[i-1] ])+f' (#{i})', color='black', fontsize=10, ha='left', va='top', 
						bbox=dict(facecolor='yellow', alpha=0.9, boxstyle='round,pad=0.3'))  # Add label
			ax.axis('off')  # Turn off axes

		plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

		# save plot
		filename = os.path.join(self.saveFolder, self.slideName + f'_thumbAbundance.jpeg')
		plt.savefig(filename, bbox_inches='tight', dpi=100)
		plt.close()

		# display plot
		RedNode = slicer.util.getNode("vtkMRMLSliceNodeRed")
		markupNodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsNode")
		for markupNode in markupNodes:
			displayNode = markupNode.GetDisplayNode()
			displayNode.SetViewNodeIDs([RedNode.GetID()])

		YellowCompNode = slicer.util.getNode("vtkMRMLSliceCompositeNodeYellow")
		YellowNode = slicer.util.getNode("vtkMRMLSliceNodeYellow")

		volumeNode = slicer.util.loadVolume(filename, {"singleFile": True})
		slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpYellowSliceView)

		YellowCompNode.SetBackgroundVolumeID(volumeNode.GetID())
		YellowNode.SetOrientation("Axial")
		slicer.util.resetSliceViews()

		return True
	
	def ViewContrastThumbnail(self):
		mz_inds = self.contrast_thumbnail_inds
		dim_y = self.dim_y
		dim_x = self.dim_x

		top_n = int(len(mz_inds)/3)

		# fig, axes = plt.subplots(3, top_n, figsize=(top_n/dim_y*dim_x*2, 3*2), gridspec_kw={'wspace': 0, 'hspace': 0})

		# for i, ax in enumerate(axes.flat):
		# 	ion_image = self.peaks_norm[:, mz_inds[i]].reshape((dim_y,dim_x),order='C')
		# 	ax.imshow(ion_image, cmap='inferno')
		# 	ax.text(0, 0, str(self.mz[ mz_inds[i] ]), color='black', fontsize=10, ha='left', va='top', 
		# 			bbox=dict(facecolor='yellow', alpha=0.9, boxstyle='round,pad=0.3'))  # Add label
		# 	ax.axis('off')  # Turn off axes

		peaks_pca = self.peaks_pca
		fig, axes = plt.subplots(3, top_n+1, figsize=((top_n+1)/dim_y*dim_x*3, 3*3), gridspec_kw={'wspace': 0, 'hspace': 0})

		for i, ax in enumerate(axes.flat):
			q, r = divmod(i, top_n+1)
			if r==0: # start each line with the PC image
				pc_image = peaks_pca[:,q].reshape((dim_y,dim_x),order='C')
				pc_image = pc_image[::2,::2]
				ax.imshow(pc_image, cmap='inferno')
				ax.text(0, 0, f'PC{q+1}', color='white', fontsize=10, ha='left', va='top', 
						bbox=dict(facecolor='black', alpha=0.9, boxstyle='round,pad=0.3'))  # Add label
			else: # ion images
				ion_index = i-1-q
				ion_image = self.peaks_norm[:, mz_inds[ion_index]].reshape((dim_y,dim_x),order='C')
				ion_image = ion_image[::2,::2]
				ax.imshow(ion_image, cmap='inferno')

				if r>(top_n/2): #label change for positive and negative loadings
					label = str(self.mz[ mz_inds[ion_index] ])+"\u2193"
					textColor = 'yellow'
					boxColor =  'black'
				else:
					label = str(self.mz[ mz_inds[ion_index] ])+"\u2191"
					textColor = 'black'
					boxColor =  'yellow'

				ax.text(0, 0, label, color=textColor, fontsize=10, ha='left', va='top', 
						bbox=dict(facecolor=boxColor, alpha=0.9, boxstyle='round,pad=0.3'))  # Add label
			
			ax.axis('off')  # Turn off axes

		plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

		# save plot
		filename = os.path.join(self.saveFolder, self.slideName + f'_thumbCont.jpeg')
		plt.savefig(filename, bbox_inches='tight', dpi=100)
		plt.close()

		# display plot
		RedNode = slicer.util.getNode("vtkMRMLSliceNodeRed")
		markupNodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsNode")
		for markupNode in markupNodes:
			displayNode = markupNode.GetDisplayNode()
			displayNode.SetViewNodeIDs([RedNode.GetID()])

		YellowCompNode = slicer.util.getNode("vtkMRMLSliceCompositeNodeYellow")
		YellowNode = slicer.util.getNode("vtkMRMLSliceNodeYellow")

		volumeNode = slicer.util.loadVolume(filename, {"singleFile": True})
		slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpYellowSliceView)

		YellowCompNode.SetBackgroundVolumeID(volumeNode.GetID())
		YellowNode.SetOrientation("Axial")
		slicer.util.resetSliceViews()

	def VisCluster(self, n_clusters):
		kmeans = KMeans(n_clusters=n_clusters, random_state=42)
		labels = kmeans.fit_predict(self.peaks_pca)  # Cluster labels for each pixel
		self.pixel_clusters = labels
		## clusters have the same color as visualization
		# centers = kmeans.cluster_centers_  # RGB values of cluster centers
		# color_image = centers[labels].reshape((self.dim_y,self.dim_x,-1),order='C')
		
		## clusters have jet colormap
		cluster_colors = cm.get_cmap('jet')(np.linspace(0, 1, n_clusters))
		cluster_colors = cluster_colors[:,:3]
		color_image = cluster_colors[labels].reshape((self.dim_y,self.dim_x,-1),order='C')

		color_image = np.expand_dims(color_image, axis=0)*255

		self.visualizationRunHelper(color_image, color_image.shape, visualization_type='cluster')

		return cluster_colors

	def dice_score(self, segmentation_mask):
		# Define thresholds
		thresholds = np.linspace(0.1, 0.9, 9)  # 9 thresholds

		# Precompute sums of the segmentation mask
		mask_sum = np.sum(segmentation_mask)  # Total pixels in the mask

		# Precompute thresholded data for all thresholds (vectorized)
		thresholded_data = np.expand_dims(self.peaks_norm, axis=2) > thresholds  # Shape: (pixels, ions, thresholds)

		# Calculate intersection and union for Dice score using einsum
		intersection = np.einsum('p, pit -> it', segmentation_mask, thresholded_data)  # Shape: (ions, thresholds)
		thresh_sum = np.sum(thresholded_data, axis=0)  # Shape: (ions, thresholds)

		dice_scores = 2 * intersection / (mask_sum + thresh_sum + 1e-10)  # Avoid division by zero

		# Keep Dice scores as (ions, thresholds) and adjust subsequent calculations accordingly
		# Find the max Dice score for each ion across thresholds
		max_dice_scores = np.max(dice_scores, axis=1)  # Shape: (ions,)
		max_threshold_indices = np.argmax(dice_scores, axis=1)  # Shape: (ions,)

		# Combine max scores and corresponding thresholds
		max_thresholds = thresholds[max_threshold_indices]

		return max_dice_scores, max_thresholds
	
	def pearson_corr(self, segmentation_mask):
		x_centered = segmentation_mask - segmentation_mask.mean()
		Y_centered = self.peaks_norm - self.peaks_norm.mean(axis=0)

		numerator = np.dot(x_centered, Y_centered)
		denominator = np.sqrt(np.sum(x_centered**2)) * np.sqrt(np.sum(Y_centered**2, axis=0))
		denominator[denominator == 0] = np.nan  # Avoid division by zero

		pearson_corrs = numerator / denominator

		return pearson_corrs

	def ViewClusterThumbnail(self, cluster_id):
		segmentation_mask = (self.pixel_clusters == cluster_id).astype(int)

		max_dice_scores, max_thresholds = self.dice_score(segmentation_mask)
		pearson_corrs = self.pearson_corr(segmentation_mask)

		sorted_indices = np.argsort(-pearson_corrs)

		# Get top 5 ions and their scores for thumbnail
		top_n = 10 
		top_ions = sorted_indices[:top_n]

		# print("Top 5 ions based on max Dice score:")
		# for rank, ion_idx in enumerate(top_ions):
		# 	print(f"Rank {rank + 1}: Ion {ion_idx}, mz {self.mz[ion_idx]}, Max Dice Score = {max_dice_scores[ion_idx]:.4f}, Threshold = {max_thresholds[ion_idx]:.1f}")

		mz_inds = top_ions
		dim_y = self.dim_y
		dim_x = self.dim_x

		fig, axes = plt.subplots(1, 1+top_n, figsize=(top_n/dim_y*dim_x*3, 1*3), gridspec_kw={'wspace': 0, 'hspace': 0})

		for i, ax in enumerate(axes.flat):
			if i==0:
				mask_image = segmentation_mask.reshape((dim_y,dim_x),order='C')
				ax.imshow(mask_image, cmap='inferno')
				ax.text(0, 0, 'cluster', color='yellow', fontsize=10, ha='left', va='top', 
						bbox=dict(facecolor='black', alpha=0.9, boxstyle='round,pad=0.3'))  # Add label
			else:
				ion_image = self.peaks_norm[:, mz_inds[i-1]].reshape((dim_y,dim_x),order='C')
				ax.imshow(ion_image, cmap='inferno')
				ax.text(0, 0, str(self.mz[ mz_inds[i-1] ]), color='black', fontsize=10, ha='left', va='top', 
						bbox=dict(facecolor='yellow', alpha=0.9, boxstyle='round,pad=0.3'))  # Add label
			ax.axis('off')  # Turn off axes

		plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

		# save plot
		filename = os.path.join(self.saveFolder, self.slideName + f'_thumbCluster.jpeg')
		plt.savefig(filename, bbox_inches='tight', dpi=100)
		plt.close()

		# display plot
		RedNode = slicer.util.getNode("vtkMRMLSliceNodeRed")
		markupNodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsNode")
		for markupNode in markupNodes:
			displayNode = markupNode.GetDisplayNode()
			displayNode.SetViewNodeIDs([RedNode.GetID()])

		YellowCompNode = slicer.util.getNode("vtkMRMLSliceCompositeNodeYellow")
		YellowNode = slicer.util.getNode("vtkMRMLSliceNodeYellow")

		volumeNode = slicer.util.loadVolume(filename, {"singleFile": True})
		slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpYellowSliceView)

		YellowCompNode.SetBackgroundVolumeID(volumeNode.GetID())
		YellowNode.SetOrientation("Axial")
		slicer.util.resetSliceViews()
		
		top_n_table = 10 
		top_ions_table = sorted_indices[:top_n_table]
		volcano_fc, volcano_pval = self.volcano_table(segmentation_mask>0, top_ions_table)

		return self.mz[top_ions_table], max_dice_scores[top_ions_table], volcano_fc, volcano_pval, pearson_corrs[top_ions_table]
	
	def volcano_table(self, mask, top_ind):
		inside_segment = self.peaks_norm[mask][:, top_ind]
		outside_segment = self.peaks_norm[~mask][:, top_ind]

		mean_inside = np.mean(inside_segment, axis=0)  # Mean for each ion inside the segment
		mean_outside = np.mean(outside_segment, axis=0)  # Mean for each ion outside the segment
		std_inside = np.std(inside_segment, axis=0, ddof=1)  # Standard deviation inside
		std_outside = np.std(outside_segment, axis=0, ddof=1)  # Standard deviation outside

		n_inside = inside_segment.shape[0]
		n_outside = outside_segment.shape[0]

		pooled_std = np.sqrt((std_inside**2 / n_inside) + (std_outside**2 / n_outside))
		t_stat = (mean_inside - mean_outside) / (pooled_std + 1e-10)  # Avoid division by zero
		_, p_values = ttest_ind(inside_segment, outside_segment, axis=0, equal_var=False)
		p_values = -np.log10(p_values+1e-300)

		fold_changes = np.log2(mean_inside / (mean_outside + 1e-10))  # Avoid division by zero

		return fold_changes, p_values
	
	def spectrum_plot(self):
		self.clear_all_plots()
		# Collect fiducial information
		fiducial_nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsFiducialNode")
		fnode_names = []
		fnode_locs = []

		for fiducial_node in fiducial_nodes:
			num_fiducials = fiducial_node.GetNumberOfControlPoints()
			for i in range(num_fiducials):
				position = [0.0, 0.0, 0.0]
				fiducial_node.GetNthControlPointPosition(i, position)
				point_name = fiducial_node.GetNthControlPointLabel(i)
				fnode_names.append(point_name)
				fnode_locs.append(self.fiducial_to_index(position))

		N = len(fnode_locs)
		if N == 0:
			print("No fiducials found.")
			return False
		print(f"Number of fiducials: {N}")

		# Create or update a plot for each fiducial
		for i, (fnode_name, fnode_loc) in enumerate(zip(fnode_names, fnode_locs)):
			fnode_ind = ind_ToFrom_sub(fnode_loc, self.dim_x)
			spec = self.peaks[fnode_ind, :]

			plotViewNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotViewNode", f"Plot{i+1}")
			plotViewNode.SetSingletonTag(f"Plot{i+1}")
			plotViewNode.SetLayoutLabel(f"Plot{i+1}")
		
			plotChartNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode", f"PlotChart{i+1}")
			plotChartNode.SetTitle(f"{fnode_name}")
			plotChartNode.SetXAxisTitle("m/z")
			plotChartNode.SetYAxisTitle("intensity")
			plotChartNode.SetLegendVisibility(False)

			# Create plot series and table
			plotSeriesNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", f"Fiducial {fnode_name}")
			tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
			table = tableNode.GetTable()

			# Populate table with data
			col_mz = vtk.vtkFloatArray()
			col_mz.SetName("m/z")
			col_intensity = vtk.vtkFloatArray()
			col_intensity.SetName("Intensity")
			for mz_value, intensity in zip(self.mz, spec):
				col_mz.InsertNextValue(mz_value)
				col_intensity.InsertNextValue(intensity)
			table.AddColumn(col_mz)
			table.AddColumn(col_intensity)
			
			ranks = self.calculate_ranks(col_intensity)
			col_label = vtk.vtkStringArray()
			col_label.SetName("Label")
			for mz_value, intensity, rank in zip(self.mz, spec, ranks):
				label = f"\nm/z: {mz_value}\nIntensity: {intensity:.2e}\nRank: {rank}"
				col_label.InsertNextValue(label)
			table.AddColumn(col_label)
			# Link data to series and chart
			plotSeriesNode.SetAndObserveTableNodeID(tableNode.GetID())
			plotSeriesNode.SetXColumnName("m/z")
			plotSeriesNode.SetYColumnName("Intensity")
			plotSeriesNode.SetLabelColumnName("Label")
			plotSeriesNode.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
			plotSeriesNode.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleNone) 
			plotSeriesNode.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleSolid)
			colour = cm.get_cmap("tab10")(i % 10)[:3]  # Get RGB values from 'tab10' colormap
			plotSeriesNode.SetColor(*colour)

			plotChartNode.SetYAxisLogScale(False)
			plotChartNode.AddAndObservePlotSeriesNodeID(plotSeriesNode.GetID())
			
			# Link chart to view
			plotViewNode.SetPlotChartNodeID(plotChartNode.GetID())

		# Update layout dynamically
		self.update_layout(N)
		
		print("Interactive plot updated with fiducials.")
		
		return True

	def calculate_ranks(self, intensity_array):
		""" Calculate ranks for intensities (1 = highest intensity) """
		# Get intensity values
		intensities = [intensity_array.GetValue(i) for i in range(intensity_array.GetNumberOfValues())]
		sorted_intensities = sorted(intensities, reverse=True)
		rank_dict = {intensity: rank + 1 for rank, intensity in enumerate(sorted_intensities)}
		ranks = [rank_dict[intensity] for intensity in intensities]
		return ranks

	def update_layout(self, N):
		layoutXML = """
		<layout type="horizontal">
			<item>
				<view class="vtkMRMLSliceNode" singletontag="Red">
					<property name="orientation" action="default">Axial</property>
					<property name="viewlabel" action="default">R</property>
					<property name="viewcolor" action="default">#F34A4A</property>
				</view>
			</item>
			<item>
				<layout type="vertical">
		"""
		for i in range(N):
			layoutXML += f"""
				<item>
					<view class="vtkMRMLPlotViewNode" singletontag="Plot{i+1}">
						<property name="viewlabel" action="default">Plot{i+1}</property>
					</view>
				</item>
			"""
		layoutXML += """
				</layout>
			</item>
		</layout>
		"""

		layoutNode = slicer.app.layoutManager().layoutLogic().GetLayoutNode()
		customLayoutId = N * 500
		layoutNode.AddLayoutDescription(customLayoutId, layoutXML)
		layoutNode.SetViewArrangement(customLayoutId)
		slicer.app.processEvents()
		
		# Clear old selections
		for i in range(slicer.app.layoutManager().plotViewCount):
			slicer.app.layoutManager().plotWidget(i).plotView().RemovePlotSelections()  
		# Remove existing connections
		for i in range(slicer.app.layoutManager().plotViewCount):
			plotView = slicer.app.layoutManager().plotWidget(i).plotView()
			try:
				plotView.disconnect("dataSelected(vtkStringArray*, vtkCollection*)", self.get_data)  # Remove previous connections
			except TypeError:
				pass
		# Connect to data selection event
		for i in range(slicer.app.layoutManager().plotViewCount):
			plotView = slicer.app.layoutManager().plotWidget(i).plotView()
			plotView.connect("dataSelected(vtkStringArray*, vtkCollection*)", self.get_data)
			#slicer.app.layoutManager().plotWidget(i).plotView().fitToContent()

	def get_data(self, data, collection):
		if collection.GetNumberOfItems() == 0:
			return
		selected_item = collection.GetItemAsObject(0)
		if selected_item is None:
			print("No valid selection.")
			return
		
		row_index = int(selected_item.GetValue(0))
		# Identify which plot widget triggered the selection
		for i in range(slicer.app.layoutManager().plotViewCount):
			plotWidget = slicer.app.layoutManager().plotWidget(i)
			plotViewNode = slicer.mrmlScene.GetNodeByID(plotWidget.mrmlPlotViewNode().GetID())
			plotChartNode = slicer.mrmlScene.GetNodeByID(plotViewNode.GetPlotChartNodeID())
			plotSeriesNodeID = plotChartNode.GetNthPlotSeriesNodeID(0)  # Get first plot series in chart
			plotSeriesNode = slicer.mrmlScene.GetNodeByID(plotSeriesNodeID)
			tableNode = slicer.mrmlScene.GetNodeByID(plotSeriesNode.GetTableNodeID())
			table = tableNode.GetTable()
			mz_value = round(float(table.GetValue(row_index, 0).ToDouble()),4)  # Column 0 = m/z values
			self.singleIonVisualization(mz_value, heatmap="Inferno")
			self.update_layout(slicer.app.layoutManager().plotViewCount)
			#plotWidget.plotView().fitToContent()
			return
		
	def clear_all_plots(self):
		"""Remove all plot nodes when no fiducials exist."""
		existingPlotViewNodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLPlotViewNode")
		existingPlotChartNodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLPlotChartNode")
		existingPlotSeriesNodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLPlotSeriesNode")
		existingPlotTableNodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLTableNode")
		for i in range(existingPlotViewNodes.GetNumberOfItems()):
			slicer.mrmlScene.RemoveNode(existingPlotViewNodes.GetItemAsObject(i))
		for i in range(existingPlotChartNodes.GetNumberOfItems()):
			slicer.mrmlScene.RemoveNode(existingPlotChartNodes.GetItemAsObject(i))
		for i in range(existingPlotSeriesNodes.GetNumberOfItems()):
			slicer.mrmlScene.RemoveNode(existingPlotSeriesNodes.GetItemAsObject(i))
		for i in range(existingPlotTableNodes.GetNumberOfItems()):
			slicer.mrmlScene.RemoveNode(existingPlotTableNodes.GetItemAsObject(i))
		slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)
		slicer.app.processEvents()
		print("Cleared all plots.")


	def fiducial_to_index(self, position):
		I = int(np.round(-position[0]))
		I = np.min( [np.max([I, 0]) , self.dim_x] )
		J = int(np.round(-position[1]))
		J = np.min( [np.max([J, 0]) , self.dim_y] )
		return(np.array([J,I]))


	# runs the single ion color channels visualization
	def multiIonVisualization(self,colors):
		selectedcolourchannels = [(color, mz) for (color, mz) in colors if mz != 'None']
				
		arrayList, arraySize = [], None

		# for all the color channels generate the image in the selected color and add the array to the 
		# array to be stacked at the end
		for i in range(len(selectedcolourchannels)):
			colour, mz = selectedcolourchannels[i]
			array = self.single_ion_display_colours(float(mz))
			array = np.transpose(array, (2, 0, 1))
			scaled = np.interp(array, (array.min(), array.max()), (0, 255))
			arraySize = scaled.shape if arraySize == None else arraySize

			stacked = np.stack((scaled,) * 3, axis=-1)
				
			if colour == 'red':
				if array.shape[0] != 1:
					array = array[49]
					array = np.expand_dims(array,axis=0)   
				stacked[:, :, :, 1] = 0
				stacked[:, :, :, 2] = 0
			elif colour == "green":
				stacked[:, :, :, 0] = 0
				stacked[:, :, :, 2] = 0
			elif colour == "blue":
				stacked[:, :, :, 0] = 0
				stacked[:, :, :, 1] = 0
			elif colour == "yellow":
				stacked[:, :, :, 2] = 0
			elif colour == "magenta":
				stacked[:, :, :, 1] = 0
			elif colour == "cyan":
				stacked[:, :, :, 0] = 0 

			arrayList.append(stacked)
			
			# adds all the channels together to visualize all at the same timee
			overlay = sum(arrayList)
		
		# runs the visualization run helper
		self.visualizationRunHelper(overlay,arraySize)

		# when worked return true
		return True
	

	# gets the segmentations, saves as an excel file
	def csvGeneration(self, filename):
		import csv
		# open the file in the write mode
		# filename = self.savenameBase + '_dataset.csv'
		f = open(filename, 'w',newline='')
		# create the csv writer
		writer = csv.writer(f)
		
		# header row
		row = ['Slide','Class','Y','X']
		row += [self.mz[i] for i in range(len(self.mz))]
		writer.writerow(row)

		# actual data rows
		# lm = slicer.mrmlScene.GetFirstNodeByName("Segmentation")
		lm = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLSegmentationNode')
		segments = lm.GetSegmentation()
		
		## template pathology classes
		# segmentation = segmentationNode.GetSegmentation()
		# segment = segmentation.GetNthSegment(0)
		# segment.SetName("NewName")
		# slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLSegmentationNode').SetName('Segmentation')
		# segmentationNode = getNode('Segmentation')
		# segmentation = segmentationNode.GetSegmentation()
		# segmentId = segmentation.GetSegmentIdBySegmentName('Segment_1')
		# segment = segmentation.GetSegment(segmentId)
		# segment.RemoveAllRepresentations()

		names, i = [], 0
		segment_id = segments.GetNthSegmentID(i)

		while segment_id:
			names += [segment_id]
			segment_id = segments.GetNthSegmentID(i + 1)
			i += 1

		for (i, name) in enumerate(names):
			segment_name = segments.GetNthSegment(i).GetName()
		 
			a = slicer.util.arrayFromSegmentBinaryLabelmap(lm, name)
			shapey = a.shape
			
			for x in range(shapey[1]):
				for y in range(shapey[2]):
					if a[0][x][y] == 1:
						row2 = []
						#################################
						row2.append(self.slideName)
						#################################
						row2.append(segment_name.lower())
						row2.append(x)
						row2.append(y)
						
						
						for i in range(len(self.mz)): 
							# row2.append(self.peaks_3D[x][y][i])
							xy = ind_ToFrom_sub([x,y], self.dim_x)
							row2.append(self.peaks[xy][i])

						writer.writerow(row2)
		f.close()
	
		df = pd.read_csv(filename)
		retstr = 'Dataset successfully created \n'
		retstr += f'File:\t \t {self.slideName}_dataset.csv \n'
		retstr += f'Number of classes:\t {len(set(df["Class"]))}\n'
		retstr += f'Total number of spectra:\t {len(df["Class"])}\n'
		# retstr += f'Spectra per class:\n'
		class_names,class_lens = np.unique(df["Class"], return_counts=1)
		for x,y in zip(class_names,class_lens):
			retstr += f'{str(y).ljust( len( str(sum(class_lens)) ) )} spectra in class {x} \n'
		return retstr

	# gets the segmentations, saves them as images to directory they are working in 
	def segmentationSave(self, savepath):
		segmentationNode = slicer.util.getNodesByClass('vtkMRMLSegmentationNode')[0]
		segmentation = segmentationNode.GetSegmentation()
		segIDs = segmentation.GetSegmentIDs()
		segNames = [segmentation.GetSegment(segID).GetName() for segID in segIDs]
		segColors = [segmentation.GetSegment(segID).GetColor() for segID in segIDs]
		for i in range(len(segIDs)):
			img = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segIDs[i])[0]
			segColor = segColors[i]

			color_img = np.zeros( (img.shape[0], img.shape[1], 3) )
			for ii in range(3):
				color_img[:,:,ii] = img * 255 * segColor[ii]
			color_img = PILImage.fromarray( np.uint8(color_img) )
			color_img.save(savepath[:-4] + '_'+ segNames[i] + '_c.png')
			
			img = PILImage.fromarray( np.uint8(img * 255) ) 
			img.save(savepath[:-4] + '_'+ segNames[i] + '.png')


	# set volume dimensions, fill with image requested, get rid of existing overlays
	def visualizationRunHelper(self,overlay,arraySize,visualization_type='multi', heatmap='Gray'):
		
		# delete all current views as we will load a new volume
		filename = f'{self.slideName}_{visualization_type}'
		if not (visualization_type.endswith('pca') or visualization_type.endswith('lda') or visualization_type.endswith('cluster')) : filename += 'ion'
		existingOverlays = slicer.util.getNodes(f'{filename}*')
		for node in existingOverlays: slicer.mrmlScene.RemoveNode(existingOverlays[node])
		
		imagesize = [arraySize[2],arraySize[1],1]
		voxelType = vtk.VTK_UNSIGNED_CHAR
		imageOrigin = [0.0, 0.0, 0.0]
		imageSpacing = [1.0, 1.0, 1.0]

		### REIMS
		if self.dim_y==1:
			imageSpacing = [1, self.REIMS_H, 1]
			# print("REIMS detected. Spacing chnaged to",imageSpacing)
		### REIMS

		imageDirections = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
		fillVoxelValue = 0

		# Create an empty image volume, filled with fillVoxelValue
		imageData = vtk.vtkImageData()
		imageData.SetDimensions(imagesize)

		imageData.AllocateScalars(voxelType, 1 if visualization_type == 'single' else 3)
		imageData.GetPointData().GetScalars().Fill(fillVoxelValue)

		volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLVectorVolumeNode", filename)
		volumeNode.SetOrigin(imageOrigin)
		volumeNode.SetSpacing(imageSpacing)
		volumeNode.SetIJKToRASDirections(imageDirections)
		volumeNode.SetAndObserveImageData(imageData)
		volumeNode.CreateDefaultDisplayNodes()
		volumeNode.CreateDefaultStorageNode()
		

		voxels = slicer.util.arrayFromVolume(volumeNode)
		voxels[:] = overlay
		volumeNode.Modified()
		displayNode = volumeNode.GetDisplayNode()
		displayNode.AutoWindowLevelOff()
		displayNode.SetWindow(240)
		displayNode.SetLevel(125)
		
		# save volume (for single-ion) or image (pca/ multi-ion)
		image = np.squeeze(overlay) if visualization_type != 'single' else overlay
		image = sitk.GetImageFromArray(image, isVector= visualization_type != 'single')
		image = sitk.Cast(image, sitk.sitkVectorUInt8) if visualization_type != 'single' else image
		sitk.WriteImage(image, os.path.join(self.saveFolder, f'{filename}.{"jpeg" if visualization_type != "single" else "nii"}'))
		
		if visualization_type != 'single': 
			slicer.util.setSliceViewerLayers(background=volumeNode, foreground=None)
		else: 
			# delete current node and reload the volume
			slicer.mrmlScene.RemoveNode(volumeNode)
			volumeNode = slicer.util.loadVolume(os.path.join(self.saveFolder, f'{filename}.nii'), {"singleFile": True})

			### REIMS
			if self.dim_y==1:
				volumeNode.GetImageData().SetSpacing(imageSpacing)
				volumeNode.Modified()
			### REIMS

			# set heatmap according to user's specification
			displayNode = volumeNode.GetDisplayNode()
			# colorNodeID = 'vtkMRMLColorTableNodeFile' + heatmap + '.txt'
			colorNodeID = slicer.util.getNode(heatmap).GetID()
			displayNode.SetAndObserveColorNodeID(colorNodeID)

		lm = slicer.app.layoutManager()
		lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)
		slicer.util.resetSliceViews()
		widget = slicer.vtkMRMLWindowLevelWidget()
		widget.SetSliceNode(slicer.util.getNode('vtkMRMLSliceNodeRed'))
		widget.SetMRMLApplicationLogic(slicer.app.applicationLogic())

		return True

	# uploads the modelling file and saves the file names into the class directory 
	def modellingFileLoad(self, filename):
		self.df = pd.read_csv(filename, dtype={"Slide": str, "Class": str})
		self.df["Class"] = self.df["Class"].str.lower()
		self.modellingFile = filename
		retstr = 'Dataset successfully loaded \n'
		retstr += f'File:                {self.modellingFile.split("/")[-1]}\n'
		retstr += f'Slides:            {len(set(self.df["Slide"]))}\n'
		retstr += f'Classes:         {len(set(self.df["Class"]))}\n'
		retstr += f'Spectra:         {len(self.df)}\n'
		retstr += f'# m/z values: {self.df.shape[1]-4}\n'
		for classname in set(self.df["Class"]):
			retstr += f'{classname}: {len(self.df.loc[self.df["Class"] == classname])}\n'
		return retstr


	def balanceData(self, X_train, y_train, track_info_train):

		balanceType = self.train_balancing

		class_names, class_counts = np.unique(y_train, return_counts=True)

		X_train_balanced, y_train_balanced, track_info_train_balanced = [], [], []

		if balanceType == 'down-sample':
			sample_per_class = int(class_counts.min())
		elif balanceType == 'up-sample':
			sample_per_class = int(class_counts.max())
		elif balanceType == 'mid-sample':
			sample_per_class = int(class_counts.mean())

		for cls in class_names:
			cls_idx = np.where(y_train == cls)[0]

			with_replace = False
			if sample_per_class>len(cls_idx):
				with_replace = True

			cls_sampled_idx = resample(cls_idx, replace=with_replace, n_samples=sample_per_class, random_state=100)
			_, a = np.unique(cls_sampled_idx, return_counts=1)
			print(len(cls_idx), np.unique(a, return_counts=1))
			
			X_train_balanced.append( X_train[cls_sampled_idx] )
			y_train_balanced.append( y_train[cls_sampled_idx] )
			track_info_train_balanced.append( track_info_train[cls_sampled_idx] )

		X_train_balanced = np.concatenate(X_train_balanced, axis=0)
		y_train_balanced = np.concatenate(y_train_balanced, axis=0)
		track_info_train_balanced = np.concatenate(track_info_train_balanced, axis=0)

		return X_train_balanced, y_train_balanced, track_info_train_balanced

	
	def runLDA(self, X_train, X_test, y_train, y_test):
		
		pca_model = PCA(n_components=0.99)
		pca_model.fit(X_train)
		print('number of PC:',pca_model.n_components_)
		X_train_pca = pca_model.transform(X_train)
		X_test_pca = pca_model.transform(X_test)

		n_class = len(np.unique(y_train))
		lda_model = LDA(n_components=n_class - 1)
		lda_model.fit(X_train_pca, y_train)
		X_train_lda = lda_model.transform(X_train_pca)
		y_train_preds = lda_model.predict(X_train_pca)
		y_train_prob = lda_model.predict_proba(X_train_pca)
		X_test_lda = lda_model.transform(X_test_pca)
		y_test_preds = lda_model.predict(X_test_pca) 
		y_test_prob = lda_model.predict_proba(X_test_pca)

		class_order = lda_model.classes_
		
		self.LDA_plot(X_train_lda, y_train)
		return y_train_preds, y_train_prob, y_test_preds, y_test_prob, class_order, [pca_model, lda_model]
		
	# def runDecisionTree(self, X_train, X_test, y_train, y_test):
	# 	model = DecisionTreeClassifier()
	# 	model.fit(X_train, y_train)
	# 	test_preds, train_preds = model.predict(X_test), model.predict(X_train)
	# 	# dir_checkpoint = '/'.join(self.modellingFile.split('/')[1:-1])
	# 	# torch.save(model, f'{dir_checkpoint}/DT_model.pt')
	# 	return train_preds, test_preds
	
	def runRF(self, X_train, X_test, y_train, y_test):
		rf_model = RandomForestClassifier()
		rf_model.fit(X_train, y_train)
		y_train_preds = rf_model.predict(X_train)
		y_train_prob = rf_model.predict_proba(X_train)
		y_test_preds = rf_model.predict(X_test)
		y_test_prob = rf_model.predict_proba(X_test)
		class_order = rf_model.classes_
		return y_train_preds, y_train_prob, y_test_preds, y_test_prob, class_order, [rf_model] 
		
	def runSVM(self, X_train, X_test, y_train, y_test):
		svm_model = LinearSVC(dual='auto')
		svm_model.fit(X_train, y_train)
		class_order = svm_model.classes_

		y_train_preds = svm_model.predict(X_train)
		y_train_prob = svm_model.decision_function(X_train)
		if len(class_order)<=2:
			y_train_prob = expit(y_train_prob)
			y_train_prob = y_train_prob.reshape(-1,1)
			y_train_prob = np.concatenate([1-y_train_prob, y_train_prob], axis=1)
		else:
			y_train_prob = softmax(y_train_prob, axis=1)

		y_test_preds = svm_model.predict(X_test)
		y_test_prob = svm_model.decision_function(X_test)
		if len(class_order)<=2:
			y_test_prob = expit(y_test_prob)
			y_test_prob = y_test_prob.reshape(-1,1)
			y_test_prob = np.concatenate([1-y_test_prob, y_test_prob], axis=1)
		else:
			y_test_prob = softmax(y_test_prob, axis=1)

		return y_train_preds, y_train_prob, y_test_preds, y_test_prob, class_order, [svm_model] 

	def runPLS(self, X_train, X_test, y_train, y_test):
		label_encoder = OneHotEncoder(sparse_output=False)
		y_train_oh = label_encoder.fit_transform( y_train.reshape(-1, 1) )
		# y_test_oh = label_encode.transform( y_test.reshape(-1, 1) )
		class_order = label_encoder.categories_[0]
		n_class = len(class_order)

		plsda = PLSRegression(n_components = n_class)
		plsda.fit(X_train, y_train_oh)
		# X_train_plsda = plsda.transform(X_train)
		y_train_prob= plsda.predict(X_train)
		y_train_preds = label_encoder.inverse_transform(y_train_prob)
		# X_test_plsda = plsda.transform(X_test)
		y_test_prob= plsda.predict(X_test)
		y_test_preds = label_encoder.inverse_transform(y_test_prob)
		return y_train_preds, y_train_prob, y_test_preds, y_test_prob, class_order, [plsda, label_encoder] 
	
	def runModel(self, modelSavename): 
		
		# clear the yellow view
		slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpYellowSliceView)
		YellowCompNode = slicer.util.getNode("vtkMRMLSliceCompositeNodeYellow")
		YellowCompNode.SetBackgroundVolumeID(None)

		# Get X and y from loaded dataframe. X=(n, mz), y=(n,)
		X, y, track_info = self.df.iloc[0:, 4:].values, self.df.iloc[0:, 1].values, self.df.iloc[0:, 0:4].values

		# # Fit imputer to handle missing values
		# imp = SimpleImputer(missing_values=np.nan, strategy='constant')
		# imp = imp.fit(X)

		# Split data according to user specifications
		if self.split == 'random':
			X_train, X_test, y_train, y_test, track_info_train, track_info_test = train_test_split(X, y, track_info, test_size=0.3, random_state=1)
			X_val, y_val = X_test, y_test
			
			# # split test set further into half for val and test if CNN
			# if self.model_type == 'CNN':
			# 	X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=1)
				
		elif self.split == 'all_train':
			X_train, X_val, X_test, y_train, y_val, y_test = X, X, X, y, y, y
			track_info_train, track_info_test = track_info, track_info
			
		elif self.split == 'custom':
			train_data = self.df.loc[self.df['Slide'].isin(self.train_cases)]
			test_data = self.df.loc[self.df['Slide'].isin(self.test_cases)]

			if len(self.test_cases)==0:
				test_data = train_data

			val_data = test_data
			# if self.model_type == 'CNN':
			# 	val_data = self.df.loc[self.df['Slide'].isin(self.val_cases)]
				
			X_train, y_train = train_data.iloc[0:, 4:].values, train_data.iloc[0:, 1].values
			X_test, y_test = test_data.iloc[0:, 4:].values, test_data.iloc[0:, 1].values
			X_val, y_val = val_data.iloc[0:, 4:].values, val_data.iloc[0:, 1].values
			track_info_train, track_info_test = train_data.iloc[0:, 0:4].values, test_data.iloc[0:, 0:4].values
			
		print(self.split)
		print(X_train.shape, X_test.shape)

		# # Apply imputer to train and test sets before fitting model
		# X_train, X_test = imp.transform(X_train), imp.transform(X_test)
		# X_val = imp.transform(X_val) if len(X_val) > 0 else X_val
		X_train = np.nan_to_num(X_train)
		X_test = np.nan_to_num(X_test)
		X_val = np.nan_to_num(X_val)

		# balancing the data
		if self.train_balancing != "None":
			X_train, y_train, track_info_train = self.balanceData(X_train, y_train, track_info_train)

		# Ion-based normalization
		ion_normalizer = MinMaxScaler()
		X_train = ion_normalizer.fit_transform(X_train)
		X_test = ion_normalizer.transform(X_test)
		X_test = np.clip(X_test, 0, 1)
		X_val = ion_normalizer.transform(X_val)
		X_val = np.clip(X_val, 0, 1)
	
		# Train the model corresponding to model_type
		if self.model_type == 'PCA-LDA':
			MODEL = self.runLDA
		if self.model_type == 'Random Forest':
			MODEL = self.runRF
		if self.model_type == 'SVM':
			MODEL = self.runSVM
		if self.model_type == 'PLS-DA':
			MODEL = self.runPLS
		
		y_train_preds, y_train_prob, y_test_preds, y_test_prob, class_order, models = MODEL(X_train, X_test,y_train, y_test)

		# Get data information to relay to the user
		filename = self.modellingFile.split('/')[-1]
		confirmation_string = f'Model successfully trained\n'
		confirmation_string += f'model type:\t{self.model_type}\n'
		confirmation_string += f'dataset:\t{filename}\n'
		confirmation_string += f'data split:\t{self.split}\n\n'

		fold_string = get_fold_info(y_train, y_test, class_order, self.split)
		# Get performance information to relay to the user
		perf_string = get_performance_info(y_train, y_train_preds, y_train_prob, 
			 								y_test, y_test_preds, y_test_prob,
											class_order, self.split)

		all_string = confirmation_string + fold_string + "\n\n" + perf_string

		# save the model
		model_pipeline = [ion_normalizer] + models
		reference_mz = np.array(self.df.columns[4:], dtype='float')

		if modelSavename!=None:
			with open(modelSavename,'wb') as f:
					pickle.dump([self.model_type, model_pipeline, class_order, reference_mz, all_string],f)

			## tracking classification results to the pixel location
			result_header = self.df.columns[0:4].tolist()+['Predicted class']+['Prob. of ' + element for element in class_order]
			result_train = np.concatenate([track_info_train, y_train_preds.reshape([-1,1]), y_train_prob], axis=1)
			pd.DataFrame(result_train, columns=result_header).to_csv(modelSavename[:-4]+'_trainTrack.csv', index=False)
			
			if self.split != 'all_train':
				result_test = np.concatenate([track_info_test, y_test_preds.reshape([-1,1]), y_test_prob], axis=1)
				pd.DataFrame(result_test, columns=result_header).to_csv(modelSavename[:-4]+'_testTrack.csv', index=False)

		return all_string
	

	def plot_latent_pca_interactive(self, plot="Class"):
		# Extract data from the dataframe
		peaks = self.df.iloc[:, 4:].values
		labels = self.df.iloc[:, 0:4].values
		peaks = np.nan_to_num(peaks)
		
		# MinMax normalize the data and apply PCA
		pca = PCA(n_components=2)
		mm1 = MinMaxScaler()
		peaks_pca = pca.fit_transform(mm1.fit_transform(peaks))
		mm2 = MinMaxScaler()
		peaks_pca = mm2.fit_transform(peaks_pca)

		# Assign label based the designated distribution
		if plot == 'Slide':
			title = 'Slide distribution'
			label = labels[:, 0]
		elif plot == 'Class':
			title = 'Class distribution'
			label = labels[:, 1]

		# Switch to the plot layout and get plot widget and view node
		layoutManager = slicer.app.layoutManager()
		layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpPlotView)
		slicer.app.processEvents()
		plotWidget = layoutManager.plotWidget(0)
		plotViewNode = plotWidget.mrmlPlotViewNode()

		# Create a new plot chart node
		plotChartNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode", f"PlotChart1")
		plotChartNode.SetTitle(title)
		plotChartNode.SetXAxisTitle("PC1")
		plotChartNode.SetYAxisTitle("PC2")
		plotChartNode.SetLegendVisibility(True)

		# Create a new table node
		tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "PCA_Table")
		table = tableNode.GetTable()

		# Create columns for the table
		col_PC1 = vtk.vtkFloatArray()
		col_PC1.SetName("PC1")
		col_PC2 = vtk.vtkFloatArray()
		col_PC2.SetName("PC2")
		col_slide = vtk.vtkStringArray()
		col_slide.SetName("Slide")
		col_class = vtk.vtkStringArray()
		col_class.SetName("Class")
		col_X = vtk.vtkFloatArray()
		col_X.SetName("X")
		col_Y = vtk.vtkFloatArray()
		col_Y.SetName("Y")
		
		# Populate the columns with data
		for i in range(len(peaks_pca)):
			col_PC1.InsertNextValue(peaks_pca[i, 0])
			col_PC2.InsertNextValue(peaks_pca[i, 1])
			col_slide.InsertNextValue(str(labels[i, 0]))
			col_class.InsertNextValue(str(labels[i, 1]))
			col_X.InsertNextValue(labels[i, 3])
			col_Y.InsertNextValue(labels[i, 2])

		table.AddColumn(col_PC1)
		table.AddColumn(col_PC2)
		table.AddColumn(col_slide)
		table.AddColumn(col_class)
		table.AddColumn(col_X)
		table.AddColumn(col_Y)

		# Split by label and create separate series
		unique_labels = np.unique(label)
		for idx, label_val in enumerate(unique_labels):
			# Create a new table node for each unique label
			filteredTableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", f"Table_{label_val}")
			filteredTable = filteredTableNode.GetTable()

			# Create columns for the filtered table
			pc1Array = vtk.vtkFloatArray()
			pc1Array.SetName("PC1")
			pc2Array = vtk.vtkFloatArray()
			pc2Array.SetName("PC2")
			labelArrayNew = vtk.vtkStringArray()
			labelArrayNew.SetName("Labels")
			
			# Populate columns with data for the filtered table
			for i in range(len(label)):
				if label[i] == label_val:
					pc1Array.InsertNextValue(peaks_pca[i, 0])
					pc2Array.InsertNextValue(peaks_pca[i, 1])
					fill = f"\nSlide: {col_slide.GetValue(i)}\nPixel Location: ({col_X.GetValue(i)}, {col_Y.GetValue(i)})\nClass: {col_class.GetValue(i)}"
					labelArrayNew.InsertNextValue(fill)

			filteredTable.AddColumn(pc1Array)
			filteredTable.AddColumn(pc2Array)
			filteredTable.AddColumn(labelArrayNew)

			# Create a new plot series node for the filtered table
			plotSeriesNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", f"{label_val}")
			plotSeriesNode.SetAndObserveTableNodeID(filteredTableNode.GetID())
			plotSeriesNode.SetXColumnName("PC1")
			plotSeriesNode.SetYColumnName("PC2")
			plotSeriesNode.SetLabelColumnName("Labels")
			plotSeriesNode.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
			plotSeriesNode.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleCross)
			plotSeriesNode.SetMarkerSize(5)
			plotSeriesNode.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleNone)
			colour = cm.get_cmap("tab10")(idx % 10)[:3]
			plotSeriesNode.SetColor(*colour)

			plotChartNode.AddAndObservePlotSeriesNodeID(plotSeriesNode.GetID())
			plotViewNode.SetPlotChartNodeID(plotChartNode.GetID())
			
		# Assign chart to plot view and force update
		plotViewNode.SetPlotChartNodeID(plotChartNode.GetID())
		plotWidget.setMRMLPlotViewNode(plotViewNode)
		slicer.app.processEvents()


	def LDA_plot(self, X_train_lda, y_train):

		if X_train_lda.shape[1]==1: #binary
			# scatter_data = np.pad(X_train_lda, [(0,0), (0,1)])
			# scatter_data = np.concatenate([X_train_lda, np.random.rand(X_train_lda.shape[0], 1)], axis=1)
			scatter_data = X_train_lda
			scatter_labels = y_train

			fig = plt.figure(figsize=(10,10))
			ax = fig.add_subplot(111)

			legend_labels = np.unique(scatter_labels)
			n_colors = len(legend_labels)
			class_colors = plt.cm.tab10(range(n_colors))

			for i in range(len(legend_labels)):
				ind = scatter_labels == legend_labels[i]
				xx = scatter_data[ind]
				ax.hist(xx, alpha=0.7, label=legend_labels[i])
			plt.legend()
			ax.set_xlabel('LDA1')

		else:
			scatter_data = X_train_lda[:,:2]
			scatter_labels = y_train

			scatter_normalizaer = MinMaxScaler()
			scatter_data = scatter_normalizaer.fit_transform( scatter_data )

			fig = plt.figure(figsize=(10,10))
			ax = fig.add_subplot(111)

			legend_labels = np.unique(scatter_labels)
			n_colors = len(legend_labels)
			if n_colors<=10:
				class_colors = plt.cm.tab10(range(n_colors))
			else:
				class_colors = cm.get_cmap('jet_r')(np.linspace(0, 1, n_colors))

			for i in range(len(legend_labels)):
		
				ind = scatter_labels == legend_labels[i]
				xx = scatter_data[ind]
				ax.scatter(xx[:,0],xx[:,1],
						color=class_colors[i],
						label=legend_labels[i], alpha=0.8)

			plt.legend()
			ax.set_xlabel('LDA1')
			ax.set_ylabel('LDA2')
			ax.set_ylim([0,1])
			ax.set_xlim([0,1])

		# save plot
		filename = self.modellingFile[:-4] + '_LDAplot.jpeg'
		plt.savefig(filename, bbox_inches='tight', dpi=600)
		plt.close()

		# display plot
		YellowCompNode = slicer.util.getNode("vtkMRMLSliceCompositeNodeYellow")
		YellowNode = slicer.util.getNode("vtkMRMLSliceNodeYellow")

		volumeNode = slicer.util.loadVolume(filename, {"singleFile": True})
		slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpYellowSliceView)

		YellowCompNode.SetBackgroundVolumeID(volumeNode.GetID())
		YellowNode.SetOrientation("Axial")
		slicer.util.resetSliceViews()

  
	def CsvLoad(self, filename):
		self.csv_processing = pd.read_csv(filename)
		self.csvFile = filename
		retstr = 'Dataset successfully loaded! \n'
		retstr += f'Dataset name:\t {filename} \n'
		retstr += self.datasetInfo(self.csv_processing)
		return retstr

	def getCsvMzList(self):
		return list(self.csv_processing.keys()[5:])

	def fileSelect(self):
		# read and display image practise in juptyter
		fileExplorer = qt.QFileDialog()
		filePaths = fileExplorer.getOpenFileNames()
		filePaths = str(filePaths)[1:-2][1:-1]
		return filePaths
	
	def HistofileSelect(self):
		fileExplorer = qt.QFileDialog()
		filePath = fileExplorer.getOpenFileName(None, "Open pathology image", "", "Image Files (*.png *.tif* *.jpg *.jpeg);;All Files (*)")
		return filePath
		
	# gets the histopath they want to upload, uploads it and puts it in the slicer view
	def loadHistopathology(self, tryer):
		import matplotlib.image as mpimg

		img=mpimg.imread(tryer)
		# plt.imsave(self.saveFolder + 'histo.jpg',img)
		# slicer.util.loadVolume(self.saveFolder + 'histo.jpg')
		plt.imsave(self.savenameBase + '_histo.jpg',img)

		slicer.util.loadVolume(self.savenameBase + '_histo.jpg', {"singleFile": True})
		os.remove(self.savenameBase + '_histo.jpg')
		# lm = slicer.app.layoutManager()
		# lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutSideBySideView)
		
		# redCompositeNode = lm.sliceWidget('Red').mrmlSliceCompositeNode()
		# tic_image_id = slicer.util.getNode(f'{self.slideName}_tic2d').GetID()
		# redCompositeNode.SetBackgroundVolumeID(tic_image_id)
		# yellowCompositeNode = lm.sliceWidget('Yellow').mrmlSliceCompositeNode()
		# histo_image_id = slicer.util.getNode(self.slideName + '_histo').GetID()
		# yellowCompositeNode.SetBackgroundVolumeID(histo_image_id) # histo
		# sliceWidget = lm.sliceWidget('Yellow').sliceLogic().GetSliceNode().SetOrientation("Axial")
		slicer.util.resetSliceViews()

	def heatmap_display(self):


		tic = np.reshape(self.peaks.sum(axis=1), [self.dim_y, self.dim_x], order='C')
		TIC = sitk.GetImageFromArray(np.transpose(tic, [0, 1]))
		
		### REIMS
		if self.dim_y==1:
			TIC.SetSpacing((1,self.REIMS_H))
			# print("REIMS detected. Spacing chnaged to",TIC.GetSpacing())
		### REIMS
		
		tic_filename = os.path.join(self.saveFolder, f'{self.slideName}_tic2d.nrrd')
		sitk.WriteImage(TIC, tic_filename)    
		slicer.util.loadVolume(tic_filename, {"singleFile": True})
		os.remove(tic_filename)
		lm = slicer.app.layoutManager()
		lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)

		return True


	def textFileSelect(self):
		fileExplorer = qt.QFileDialog()
		# filePaths = fileExplorer.getOpenFileNames(None, "Open DESI text file", "", "Text Files (*.txt);;All Files (*)")
		# data_path_temp = filePaths[0]
		filePaths = fileExplorer.getOpenFileName(None, "Import MSI data", "", "Structured CSV (*.csv);;Hierarchical HDF5 (*.h5);;DESI Text (*.txt);;All Files (*)")
		data_path_temp = filePaths
		slide_name = data_path_temp.split('/')[-1]
		lengthy = len(slide_name)
		data_path = data_path_temp[:-lengthy]
		return data_path, slide_name

	def REIMSSelect(self):
		fileExplorer = qt.QFileDialog()
		filePaths = fileExplorer.getOpenFileName(None, "Open REIMS CSV file", "", "CSV Files (*.csv);;All Files (*)")
		data_path_temp = filePaths
		slide_name = data_path_temp.split('/')[-1]
		lengthy = len(slide_name)
		data_path = data_path_temp[:-lengthy]
		return data_path, slide_name

	def REIMSLoad(self, name):

		df = pd.read_csv(name)
		peak_start_col = 5
		mz = np.array(df.columns[peak_start_col:], dtype='float') 
		peaks = df[df.columns[peak_start_col:]].values
		dim_y = 1
		dim_x = len(peaks)

		# dim_y = 50
		# peaks = np.tile(peaks, (dim_y, 1))

		#sio.savemat(data_path+slide_name[:-4]+'.mat', dict(peaks=peaks, mz=mz, dim_y=dim_y, dim_x=dim_x))
		slide_name = name.split('/')[-1]
		data_path = name[:len(name)-len(slide_name)]
		## individual independent slide analysis, saves the name as a matlab file
		# slide_name = slide_name[:-4] + '.mat'
		#file = os.path.join(data_path,slide_name)
		#data = sio.loadmat(file)
				
		self.peaks = peaks
		self.mz = mz
		self.dim_y = dim_y
		self.dim_x = dim_x
		# self.peaks_3D = peaks.reshape((self.dim_y,self.dim_x,-1),order='C')
			
		# save path for pca image
		self.saveFolder = data_path
		self.slideName = slide_name[:-4]
		self.savenameBase = self.saveFolder+self.slideName
		print(self.savenameBase)


		# add each value
		# self.selectedmz = []
		# for i in range(len(mz)):
		# 	self.selectedmz.append(mz[i])

		return True

	def loadPipeline(self, fileName):
		## load model/mz_ref
		with open(fileName,"rb") as f:
			model_type, pipeline, class_order, reference_mz, model_report = pickle.load(f)
		self.Dmodel_type = 	model_type
		self.Dpipeline = pipeline
		self.Dclass_order = class_order
		self.DmzRef = reference_mz
		self.Dreport = model_report
		# print(model_report.split('\n'))
		name = fileName.split('/')[-1]
		retstr = 'Model pipeline successfully loaded\n'
		retstr += f'pipeline name:\t{ name }\n'
		retstr += '\n'.join(model_report.split('\n')[1:4])+'\n'
		# retstr += f'model type:\t{model_type}\n'
		retstr += f'classes:\n'
		for x in class_order:
			retstr += f'     {x}\n'

		retstr += f'\nnumber of m/z:\t{len(reference_mz)}\n'
		retstr += f'range of m/z:\t{reference_mz.min()} to {reference_mz.max()}\n'

		return retstr

	# Reads in the text file and converts it to a numpy array and saves
	def textFileLoad(self, name):

		slide_name = os.path.basename(name)
		data_path = os.path.dirname(name)
		data_extension = os.path.splitext(slide_name)[1].lower()

		if data_extension == '.txt':
			[peaks, mz, dim_y, dim_x] = self.DESI_txt2numpy(name)
		elif data_extension == '.csv':
			[peaks, mz, dim_y, dim_x] = self.MSI_csv2numpy(name)
		elif data_extension == '.h5':
			[peaks, mz, dim_y, dim_x] = self.MSI_h52numpy(name)
		else:
			pass
		
		self.peaks = peaks
		self.mz = mz
		self.dim_y = dim_y
		self.dim_x = dim_x
		# self.peaks_3D = peaks.reshape((self.dim_y,self.dim_x,-1),order='C')
			
		# save path for pca image
		self.saveFolder = data_path
		self.slideName = os.path.splitext(slide_name)[0]
		self.savenameBase = os.path.splitext(name)[0]

		# add each value
		# self.selectedmz = []
		# for i in range(len(mz)):
		# 	self.selectedmz.append(mz[i])

		return True

	# def getSelectedMz(self):
	# 	return self.selectedmz
		
	 # Define function to align peaks and merge csv file 
	def batch_peak_alignment(self, files, csv_save_name):
		""""
		OBJECTIVE:
		to align m/z values from muliple MSI slides
		DESCRIPTION:
		the finction will receive the location of the folder that contains all MSI csv data for alignment and svae the aligned csv in the same folder
		INPUTS:
		data_dir - the name of the folder that conains the individual MSI csv files 
		csv_save_name - the name of the final aligned file
		AUTHOR:
		@Moon
		"""

		## list the csv files
		for file in files: 
			print(file)
		## read the files
		print('reading the csv files...')
		#################################
		peak_start_col = 4 # all info other than peak intensities
		#################################
				
		mz, peaks, labels = [],[],[]
		for file in files:
			df = pd.read_csv(file)

			mz.append( np.array(df.columns[peak_start_col:], dtype='float') )
			peaks.append( df[df.columns[peak_start_col:]].values )
			labels.append( df[df.columns[0:peak_start_col]].values )

			labels_headers = df.columns[0:peak_start_col]
			# print(labels_headers)

		mz_all = np.sort(np.concatenate(mz)).astype('float64')
		
		## cluster m/z values that are close to eachother
		print('accumulating m/z values...')
		mz_bandwidth = 0.01 #std of the estimated mz clusters
		abundance_threshold = 0.4 #eliminate the least abundant peaks between slides
		mz_resolution = 0.01 # m/z resolution for peak clustering
		nsig = 4 # number of std in kernel width

		# subband_start = 50
		# subband_stop = 1200 
		subband_start = np.round(mz_all.min()-nsig*mz_bandwidth, -1)-10  # 50 m/z range minimum
		subband_stop = np.round(mz_all.max()+nsig*mz_bandwidth, -1)+10  # 1200 m/z range maximum
		
		# generate sample Gaussian kernel
		x = np.arange(-nsig*mz_bandwidth,nsig*mz_bandwidth+1e-5,mz_resolution)
		x = np.round(x, decimals=4)
		gkerbel = np.exp(-0.5*(x/mz_bandwidth)**2)

		mz_grid = np.arange(subband_start-1,subband_stop+1,mz_resolution)
		mz_grid = np.round(mz_grid, decimals=4)

		# find the kernel density distribution of all m/z values
		kde_grid = np.zeros(mz_grid.shape)
		width = int(len(gkerbel)/2)
		for mzi in mz_all:
			ind = np.argmin(np.abs(mz_grid-mzi))
			kde_grid[ind-width:ind+width+1]+=gkerbel # 3x faster than convolution 

		# find the peak values as the selected m/z
		locs, _ = find_peaks(kde_grid, height=0.5)

		selected_mz = mz_grid[locs]
		num_mz = len(selected_mz)

		## estimate the contribution of each slide in selected m/z
		print('estimating the contribution of each csv...')
		n_mz = len(selected_mz)
		n_csv = len(mz)
		new_mz = []

		for i in range(n_csv):
			x=[]
			for j in range(n_mz):
				x.append([])
			new_mz.append(x)    

		for i,old_mz in enumerate(mz):
			for j,x in enumerate(old_mz):
				diff = np.abs(selected_mz-x)
				ind = diff.argmin()
				new_mz[i][ind].append(x)

		mz_df = pd.DataFrame(new_mz, columns=selected_mz)
		
		# eliminate the least repeated mz 
		num_sample_per_slide = np.array([x.shape[0] for x in peaks]).T

		num_nonnan_mz = []
		for j in range(n_mz):
			x=0
			for i in range(n_csv):
				x += ( len(new_mz[i][j])>0 )*num_sample_per_slide[i]
			num_nonnan_mz.append(x)

		nonnan_threshold = int(abundance_threshold*np.sum(num_sample_per_slide))

		kept_mz_ind = np.array(num_nonnan_mz) > nonnan_threshold
		n_mz = kept_mz_ind.sum()
		print('number of selected m/z:', n_mz)

		new_mz_df = mz_df[mz_df.columns[kept_mz_ind]]
		
		# eliminate multiple m/z values
		final_mz_df = pd.DataFrame(np.nan*np.ones([n_csv,n_mz]), columns=new_mz_df.columns)
		for i in range(n_csv):
			mz_old = mz[i]
			pmean = peaks[i].mean(axis=0)/peaks[i].mean(axis=0).max()
			for j in range(n_mz):
				mz_cell = new_mz_df.loc[i, new_mz_df.columns[j]]
				if len(mz_cell)==0:
					pass
				elif len(mz_cell)==1:
					final_mz_df.loc[i, final_mz_df.columns[j]] = mz_cell[0]
				else:
					ind = [np.abs(x-mz_old).argmin() for x in mz_cell]
					i_abundant = pmean[ind].argmax()
					final_mz_df.loc[i, final_mz_df.columns[j]] = mz_cell[i_abundant]
								

		# correct the selected mz
		mz_list = final_mz_df.to_numpy()
		mz_selected_old = final_mz_df.columns
		corrected_mz = []
		for i in range( n_mz ):
			col = mz_list[:,i]
			col = col[~np.isnan(col)]
			min_i = np.abs(col-mz_selected_old[i]).argmin()
			corrected_mz.append( col[min_i] )

		corrected_mz = np.array(corrected_mz)
		final_mz_df.columns = corrected_mz

		# add label and save the m/z alignment results
		labeled_mz_df = final_mz_df.copy()
		#################################
		# ll = [labels[i][0][1] for i in range(n_csv)]
		ll = [labels[i][0][0] for i in range(n_csv)]
		#################################
		labeled_mz_df.insert(0, 'selected m/z', ll)

		print('saving the aligned m/z csv...')
		labeled_mz_df.to_csv(csv_save_name[:-4]+'_MZLIST.csv', index=False)
		
		# align peaks
		aligned_peaks = []
		for i in range(n_csv):
			mz_old = mz[i]
			newp = np.nan*np.ones([len(peaks[i]),n_mz])
			for j,col in enumerate(corrected_mz):
				mz_val = final_mz_df[col][i]
				if ~np.isnan(mz_val):
					p_ind = np.where(mz_old == mz_val)[0]
					newp[:,j] = peaks[i][:,p_ind].flat
			aligned_peaks.append(newp)

		del peaks
		aligned_peaks = np.concatenate(aligned_peaks, axis=0)
		aligned_peaks_labels = np.concatenate(labels, axis=0)

		# add label and save aligned peaks
		aligned_peaks = np.concatenate( [aligned_peaks_labels, aligned_peaks] , axis=1)
		csv_column = labels_headers.tolist()+list(corrected_mz)
		df = pd.DataFrame(aligned_peaks, columns=csv_column)
		print('saving the aligned peaks csv...')
		df.to_csv(csv_save_name, index=False)
		print('alignment done!')

		### report information
		retstr = 'Dataset successfully aligned! \n'
		retstr += f'Aligned dataset:\t {csv_save_name} \n'
		retstr += self.datasetInfo(df)
		return retstr


	# generates and displays the pca image
	def partial_pca_display(self, start_pos, end_pos, extend=False):
		
		dim_y, dim_x = self.dim_y,self.dim_x

		min_y, max_y = start_pos[0], end_pos[0]
		min_x, max_x = start_pos[1], end_pos[1]

		local_peaks_ind = np.zeros((dim_y, dim_x), dtype='bool')
		local_peaks_ind[min_y:max_y, min_x:max_x] = True
		local_peaks_ind = local_peaks_ind.reshape((dim_y*dim_x,),order='C')

		local_peaks = self.peaks_norm[local_peaks_ind]
		local_pca = PCA(n_components=3)
		local_scaler = MinMaxScaler()
		local_peaks_pca = local_pca.fit_transform( local_scaler.fit_transform( local_peaks ) )
		post_scaler = MinMaxScaler()
		post_scaler.fit(local_peaks_pca)

		if extend:
			local_peaks_pca = local_pca.transform( local_scaler.transform( self.peaks_norm ) )
			# local_peaks_pca = post_scaler.fit_transform( local_peaks_pca ) # the extended and non-extended result for the roi region would be different
			local_peaks_pca = post_scaler.transform( local_peaks_pca )
			local_peaks_pca = np.clip(local_peaks_pca, 0,1)
			local_pca_image = local_peaks_pca.reshape((dim_y,dim_x,-1),order='C')
		else:
			local_peaks_pca = post_scaler.transform( local_peaks_pca )
			local_peaks_pca = np.clip(local_peaks_pca, 0,1)
			local_pca_image = self.peaks_pca.copy()
			local_pca_image[local_peaks_ind] = local_peaks_pca
			local_pca_image = local_pca_image.reshape((dim_y,dim_x,-1),order='C')
		
		local_pca_image = np.expand_dims(local_pca_image, axis=0)*255

		self.visualizationRunHelper(local_pca_image, local_pca_image.shape, visualization_type='local_pca')
		
		self.peaks_pca = local_peaks_pca
		self.lastPCA = local_pca
		return True


	def roi_pca_display(self, mask, extend=False):
		
		dim_y, dim_x = self.dim_y,self.dim_x

		local_peaks_ind = mask>0
		local_peaks_ind = local_peaks_ind.reshape((dim_y*dim_x,),order='C')

		local_peaks = self.peaks_norm[local_peaks_ind]
		local_pca = PCA(n_components=3)
		local_scaler = MinMaxScaler()
		local_peaks_pca = local_pca.fit_transform( local_scaler.fit_transform( local_peaks ) )
		post_scaler = MinMaxScaler()
		post_scaler.fit(local_peaks_pca)

		if extend:
			local_peaks_pca = local_pca.transform( local_scaler.transform( self.peaks_norm ) )
			# local_peaks_pca = post_scaler.fit_transform( local_peaks_pca ) # the extended and non-extended result for the roi region would be different
			local_peaks_pca = post_scaler.transform( local_peaks_pca )
			local_peaks_pca = np.clip(local_peaks_pca, 0,1)
			local_pca_image = local_peaks_pca.reshape((dim_y,dim_x,-1),order='C')
		else:
			local_peaks_pca = post_scaler.transform( local_peaks_pca )
			local_peaks_pca = np.clip(local_peaks_pca, 0,1)
			# local_pca_image = self.peaks_pca.copy()
			local_pca_image = np.zeros((dim_y*dim_x,3))
			local_pca_image[local_peaks_ind] = local_peaks_pca
			local_pca_image = local_pca_image.reshape((dim_y,dim_x,-1),order='C')
		
		local_pca_image = np.expand_dims(local_pca_image, axis=0)*255

		self.visualizationRunHelper(local_pca_image, local_pca_image.shape, visualization_type='roi_pca')
		
		self.lastPCA = local_pca
		self.peaks_pca = local_peaks_pca
		return True
	

	def roi_lda_display(self, mask, extend=False):
		
		dim_y, dim_x = self.dim_y,self.dim_x

		local_peaks_ind = mask>0
		local_peaks_ind = local_peaks_ind.reshape((dim_y*dim_x,),order='C')

		local_labels = mask.reshape((dim_y*dim_x,),order='C')
		local_labels = local_labels[local_peaks_ind]

		local_peaks = self.peaks_norm[local_peaks_ind]
		local_pca = PCA(n_components=0.99)
		local_scaler = MinMaxScaler()
		local_peaks_pca = local_pca.fit_transform( local_scaler.fit_transform( local_peaks ) )
		local_lda = LDA()
		local_lda.fit(local_peaks_pca, local_labels)
		print(local_pca.n_components_)

		if extend:
			local_peaks_pca = local_pca.transform( local_scaler.transform( self.peaks_norm ) )
			local_peaks_lda = local_lda.predict(local_peaks_pca)
			local_lda_image = local_peaks_lda.reshape((dim_y,dim_x),order='C')
		else:
			local_peaks_lda = local_lda.predict(local_peaks_pca)
			local_lda_image = np.zeros((dim_y*dim_x,))
			local_lda_image[local_peaks_ind] = local_peaks_lda
			local_lda_image = local_lda_image.reshape((dim_y,dim_x),order='C')

		local_lda_image = local_lda_image.astype("int")
		segmentationNode = slicer.util.getNodesByClass('vtkMRMLSegmentationNode')[0]
		segmentColors = self.getSegmentColors(segmentationNode)
		local_lda_image = self.createColoredArray(local_lda_image, segmentColors)

		local_lda_image = np.expand_dims(local_lda_image, axis=0)*255

		self.visualizationRunHelper(local_lda_image, local_lda_image.shape, visualization_type='roi_lda')
		

	def getSegmentColors(self, segmentationNode):
		segmentation = segmentationNode.GetSegmentation()
		segmentColors = {}

		for i in range(segmentation.GetNumberOfSegments()):
			segmentID = segmentation.GetNthSegmentID(i)
			segment = segmentation.GetSegment(segmentID)
			color = segment.GetColor()  # Returns a tuple (R, G, B)
			# segmentColors[i+1] = color 
			segmentColors[i+1] = tuple([0.9*channel for channel in color])

		return segmentColors

	def createColoredArray(self, labelArray, segmentColors):
		# Initialize an empty array with shape M x N x 3
		coloredArray = np.zeros((labelArray.shape[0], labelArray.shape[1], 3))

		# Replace each label with its corresponding color
		for label, color in segmentColors.items():
			coloredArray[labelArray == label] = color

		return coloredArray


	def numpyArrayToSlicerLabelMap(self, numpyArray, nodeName, ijkToRASMatrix):
		# Ensure the array is in Fortran order (column-major order)
		if not numpyArray.flags['F_CONTIGUOUS']:
			numpyArray = np.asfortranarray(numpyArray)

		# Convert numpy array to VTK array
		vtkArray = numpy_support.numpy_to_vtk(num_array=numpyArray.ravel(order='F'), deep=True, array_type=vtk.VTK_INT)

		# Create a vtkImageData object and set the VTK array as its scalars
		imageData = vtk.vtkImageData()
		imageData.SetDimensions(numpyArray.shape)
		imageData.GetPointData().SetScalars(vtkArray)

		# Create a new label map volume node
		labelMapVolumeNode = slicer.vtkMRMLLabelMapVolumeNode()
		labelMapVolumeNode.SetName(nodeName)
		labelMapVolumeNode.SetAndObserveImageData(imageData)

		# Apply the IJK to RAS matrix
		vtkMatrix = vtk.vtkMatrix4x4()
		for i in range(4):
			for j in range(4):
				vtkMatrix.SetElement(i, j, ijkToRASMatrix[i, j])
		labelMapVolumeNode.SetIJKToRASMatrix(vtkMatrix)

		# Add the label map volume node to the Slicer scene
		slicer.mrmlScene.AddNode(labelMapVolumeNode)

		return labelMapVolumeNode


	def createCustomColorTable(self, segmentationNode):
		# Create a new color table
		colorTableNode = slicer.vtkMRMLColorTableNode()
		colorTableNode.SetTypeToUser()
		slicer.mrmlScene.AddNode(colorTableNode)

		segmentation = segmentationNode.GetSegmentation()
		numberOfSegments = segmentation.GetNumberOfSegments()

		# Set the number of colors in the table
		colorTableNode.SetNumberOfColors(numberOfSegments + 1)
		colorTableNode.SetColor(0, "Background", 0, 0, 0, 0)  # Background color

		# Add segment colors to the table
		for i in range(numberOfSegments):
			segmentID = segmentation.GetNthSegmentID(i)
			segment = segmentation.GetSegment(segmentID)
			color = segment.GetColor()
			colorTableNode.SetColor(i + 1, segmentID, color[0], color[1], color[2], 1)

		return colorTableNode

	def applyColormapToLabelMap(self, labelMapNode, colorTableNode):
		# Apply the color table to the label map volume node
		labelMapDisplayNode = slicer.vtkMRMLLabelMapVolumeDisplayNode()
		slicer.mrmlScene.AddNode(labelMapDisplayNode)
		labelMapDisplayNode.SetAndObserveColorNodeID(colorTableNode.GetID())
		labelMapNode.SetAndObserveDisplayNodeID(labelMapDisplayNode.GetID())

		# Update the view
		slicer.app.processEvents()


	def set_split(self, split):
		self.split = split

	def custom_split(self):
		self.set_split('custom')

		#split_names = self.df['name'].str.split('_')
		#all_case_num = [case_num for [case_num, label, burn_num] in split_names]
		#unique_cases = list(set(all_case_num))
		
		unique_cases = list(set(self.df['Slide']))
		return unique_cases

	def update_test_cases(self, cases):
		self.test_cases = cases

	def update_train_cases(self, cases):
		self.train_cases = cases
	
	def update_val_cases(self, cases):
		self.val_cases = cases

	def getDataInformation(self):
		infostr = f'{self.slideName} \n'
		infostr += f'spatial:\t {self.dim_y} x {self.dim_x} pixels \n'
		infostr += f'spectra:\t {self.dim_y*self.dim_x} \n'
		infostr += f'm/z per pixel:\t {len(self.mz)} \n'
		infostr += f'm/z range: \t {self.mz.min()} - {self.mz.max()} \n'
		return infostr
	
	def getREIMSInfo(self):
		infostr = f'Slide successfully loaded \n'
		infostr += f'File name:\t \t {self.slideName} \n'
		infostr += f'Total number of spectra:\t {self.dim_x} \n'
		infostr += f'Number of m/z per spectrum:\t {len(self.mz)} \n'
		return infostr
	
	def datasetInfo(self, df):
		retstr = f'Number of slides:\t {len(set(df["Slide"]))}\n'
		retstr += f'Number of classes:\t {len(set(df["Class"]))}\n'
		retstr += f'Total number of spectra:\t {len(df)}\n'
		class_names,class_lens = np.unique(df["Class"], return_counts=1)
		for x,y in zip(class_names,class_lens):
			retstr += f'   {str(y)}\t in class\t {x} \n'
		return retstr

def ind_ToFrom_sub(X, dim_x):
    if isinstance(X, int):
        ind = X
        i = ind // dim_x
        j = ind % dim_x
        res = [i, j]
    elif len(X)==2:
        i, j = X
        res = i * dim_x + j
    return res