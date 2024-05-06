from cProfile import label
from lib2to3.refactor import get_fixers_from_package
from math import pi
import os
import SimpleITK as sitk
from pyexpat import model
import unittest
import logging
import vtk, qt, ctk, slicer
from vtk.util import numpy_support

# Does all of the imports
try:
		import matplotlib
		import matplotlib.pyplot as plt
		import matplotlib.cm as cm
except ModuleNotFoundError:
		import pip

		slicer.util.pip_install("matplotlib")
		import matplotlib
		import matplotlib.pyplot as plt
		import matplotlib.cm as cm

## fix Mac crash
matplotlib.use('Agg')

try:
		import cv2
except ModuleNotFoundError:
		import pip

		slicer.util.pip_install("opencv-python")
		import cv2


try:
		import sklearn
		from sklearn.decomposition import PCA
		from sklearn.preprocessing import LabelEncoder
except ModuleNotFoundError:
		import pip

		slicer.util.pip_install("scikit-learn")
		import sklearn
		from sklearn.decomposition import PCA
		from sklearn.preprocessing import LabelEncoder

try:
		import pandas as pd
except ModuleNotFoundError:
		import pip

		slicer.util.pip_install("pandas")
		import pandas as pd

# try:
# 		import torch
# except ModuleNotFoundError:
# 		import pip

# 		slicer.util.pip_install("torch")
# 		import torch

# try:
# 		import logging
# except ModuleNotFoundError:
# 		import pip

# 		slicer.util.pip_install("logging")
# 		import logging
		
# try:
# 		import torchmetrics
# except ModuleNotFoundError:
# 		import pip

# 		slicer.util.pip_install("torchmetrics")
# 		import torchmetrics
  
try:
		from tqdm import tqdm
except ModuleNotFoundError:
		import pip

		slicer.util.pip_install("tqdm")
		from tqdm import tqdm
		
import numpy as np

import pdb

from SlicerDESILib.Utils import *
# from SlicerDESILib.CNN import CNN
# from SlicerDESILib.Dataset import CustomDataset
# from SlicerDESILib.CNNUtils import train, test

from slicer.ScriptedLoadableModule import *

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.cross_decomposition import PLSRegression, PLSCanonical
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.utils import resample
from scipy.special import softmax, expit

from mpl_toolkits.axes_grid1 import make_axes_locatable

import pickle

class SlicerDESILogic(ScriptedLoadableModuleLogic):
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
		self.peaks_3D = None
		self.modellingFile = None
		self.split = 'random'
		self.test_cases = set()
		self.train_cases = set()
		self.val_cases = set()
		self.selectedmz = []
		self.model_type = None
		self.train_balancing = 'None'
		#self.volume = None
		self.CNNHyperparameters = {}
		self.REIMS_H = 300
		

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

			# jet_colors = cm.get_cmap('jet')(np.linspace(0, 1, 251))
			# black_colors = np.zeros((5, 4))
			# black_colors[:, 3] = 1
			# cmap_custom = cm.colors.ListedColormap( np.vstack((black_colors, jet_colors)) )
			# plt.figure(figsize=[6,6])
			# plt.imshow(roi_reconstruct_num.T, cmap=cmap_custom, 
			# 		   interpolation=None, vmin=0, vmax=len(class2num.classes_))
			# plt.axis('off')
			# cbar = plt.colorbar(ticks=np.arange(len(class2num.classes_))+1)
			# cbar.ax.set_yticklabels(class2num.classes_)

			save_folder, save_suffix = image_save_info

			# print(save_folder)
			# print(labels_single[0,0])
			# print(save_suffix)

			save_name = os.path.join(save_folder,str(labels_single[0,0])+save_suffix)
			# plt.savefig(f"{'/'.join(self.csvFile.split('/')[:-1])}/{save_name}.png", bbox_inches='tight', dpi=300)
			#plt.show()
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

		# plt.imshow(preds_img, cmap=cmap_custom, vmin=-0.5, vmax=n_colors-0.5)
		# plt.axis('off')
		# cbar = plt.colorbar(ticks=np.arange(n_colors))
		# h = cbar.ax.set_yticklabels(self.Dclass_order)

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

		return True
	
	# generates the single ion image for the m/z value specified
	def single_ion_display_colours(self, mz_r):
		# generates and displays the single ion image
		ch_r = self.selectedmz.index(mz_r)
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
	
	def spectrum_plot(self):
		fiducial_nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsFiducialNode")
		fnode_names = []
		fnode_locs = []
		for fiducial_node in fiducial_nodes:
			num_fiducials = fiducial_node.GetNumberOfControlPoints ()
			for i in range(num_fiducials):
				position = [0.0, 0.0, 0.0]
				fiducial_node.GetNthControlPointPosition(i, position)
				point_name = fiducial_node.GetNthControlPointLabel(i)
				fnode_names.append(point_name)
				fnode_locs.append( self.fiducial_to_index(position) )
				# print(f"  Fiducial {i+1} named {point_name} position:", position, self.fiducial_to_index(position))
		# print(self.dim_y, self.dim_x)
		N = len(fnode_locs)
		plt.figure(figsize=(10,5))
		for i in range(N):
			fnode_name = fnode_names[i]
			fnode_loc = fnode_locs[i]
			plt.subplot(N,1,i+1)
			markerline, stemlines, baseline = plt.stem(self.mz, self.peaks_3D[fnode_loc[0],fnode_loc[1],:], linefmt='C'+str(i), markerfmt=" ", basefmt='C'+str(i))
			plt.setp(stemlines, linewidth=2)
			plt.legend( ['{} located at {}, {}'.format(fnode_name,fnode_loc[0],fnode_loc[1])] )
			plt.xlim([self.mz.min(), self.mz.max()])
			plt.ylim(bottom=0)
			if (i+1)==N:
				plt.xticks(np.arange(self.mz.min(),self.mz.max(),50))
			else:
				plt.xticks(np.arange(self.mz.min(),self.mz.max(),50), labels=[])
			plt.grid(True, 'both', linestyle='--')

		plt.xlabel('mass to chatge ratio')
		plt.ylabel('intensity')
		plt.savefig(self.savenameBase + '_spectra.jpeg', bbox_inches='tight', dpi=600)
		plt.close()
		
		RedCompNode = slicer.util.getNode("vtkMRMLSliceCompositeNodeRed")
		RedNode = slicer.util.getNode("vtkMRMLSliceNodeRed")
		YellowCompNode = slicer.util.getNode("vtkMRMLSliceCompositeNodeYellow")
		YellowNode = slicer.util.getNode("vtkMRMLSliceNodeYellow")

		VolumeIDonRed = RedCompNode.GetBackgroundVolumeID()

		volumeNode = slicer.util.loadVolume(self.savenameBase + '_spectra.jpeg', {"singleFile": True})

		slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutSideBySideView)
		RedCompNode.SetBackgroundVolumeID(VolumeIDonRed)
		YellowCompNode.SetBackgroundVolumeID(volumeNode.GetID())
		YellowNode.SetOrientation("Axial")
		slicer.util.resetSliceViews()
		
		markupNodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsNode")
		for markupNode in markupNodes:
			displayNode = markupNode.GetDisplayNode()
			displayNode.SetViewNodeIDs([RedNode.GetID()])

		return True

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
		row = ['Slide','Class','X','Y']
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
							row2.append(self.peaks_3D[x][y][i])
						
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
		for i in range(len(segIDs)):
			img = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segIDs[i])
			img = 255 * img.transpose(1,2,0)
			cv2.imwrite(savepath[:-4] + '_'+ segNames[i] + '.png',img)
	# def segmentationSave(self, savepath):
	# 	row = ['Class','X','Y']
	# 	row += [self.mz[i] for i in range(len(self.mz))]

	#  # actual data rows
	# 	lm = slicer.mrmlScene.GetFirstNodeByName("Segmentation")
	# 	segments = lm.GetSegmentation()
		
	# 	names, i = [], 0
	# 	segment_id = segments.GetNthSegmentID(i)

	# 	while segment_id:
	# 		names += [segment_id]
	# 		segment_id = lm.GetSegmentation().GetNthSegmentID(i + 1)
	# 		i += 1
				
	# 	count = 1
	# 	for (i, name) in enumerate(names):
	# 		a = slicer.util.arrayFromSegmentBinaryLabelmap(lm, name)
	# 		#################################
	# 		# changed = a.transpose(1,2,0)
	# 		changed = 255 * a.transpose(1,2,0)
	# 		#################################
	# 		# cv2.imwrite(self.saveFolder+ self.slideName + '_'+ name + '.png',changed)
	# 		cv2.imwrite(savepath[:-4] + '_'+ name + '.png',changed)
	# 		count = count + 1

	# set volume dimensions, fill with image requested, get rid of existing overlays
	def visualizationRunHelper(self,overlay,arraySize,visualization_type='multi', heatmap='Gray'):
		
		# delete all current views as we will load a new volume
		filename = f'{self.slideName}_{visualization_type}'
		if not (visualization_type.endswith('pca') or visualization_type.endswith('lda')) : filename += 'ion'
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
		self.df = pd.read_csv(filename)
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


	def setCNNHyperparameters(self, parameters):
		self.CNNHyperparameters = parameters

	def runCNN(self, classnames, X_train, X_val, X_test, y_train, y_val, y_test):
		"""
		Trains a CNN, saves the model weights, and plots the training curves.
		
		Args:
				classnames:
				X_train: 
				X_test: 
				y_train:
				y_test:

		Returns:
				train_predictions: Predictions on the training set.
				test_predictions: Predictions on the test set.

		Author:
				Ayesha Syeda
		"""
				
		# Hyperparameters
		batch_size = self.CNNHyperparameters['batch_size']
		epochs = 200
		learning_rate = self.CNNHyperparameters['lr']
		actual_epochs_ran = epochs
		
		# Define early stopping parameters
		patience = 5  # Number of epochs to wait for improvement
		best_loss = float('inf')
		early_stopping_counter = 0
		model_saved = False
		
		dir_checkpoint = '/'.join(self.modellingFile.split('/')[1:-1])
		
		# Keep track of performance values for each set
		train_losses, train_aurocs, train_accs = [], [], []
		test_losses, test_aurocs, test_accs = [], [], []
		val_losses, val_aurocs, val_accs = [], [], []
		
		# Imports
		import torch
		from sklearn.preprocessing import LabelEncoder
		
		# Use GPU if available
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		use_cuda = torch.cuda.is_available()
		
		# Create label mapping
		label_encoder = LabelEncoder()
		label_encoder.fit(classnames)
		y_train_numeric = label_encoder.transform(y_train)
		y_val_numeric = label_encoder.transform(y_val)
		y_test_numeric = label_encoder.transform(y_test)
		
		# Create datasets and use to create dataloaders
		train_dataset = CustomDataset(X_train, y_train_numeric)
		val_dataset = CustomDataset(X_val, y_val_numeric)
		test_dataset = CustomDataset(X_test, y_test_numeric)
		trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
		testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
		
		# Create loss weights
		weights = torch.Tensor([np.count_nonzero(y_train_numeric == i) for i in range(len(classnames))])
		weights = 1/weights
		weights = weights / weights.sum()
		weights = torch.FloatTensor(weights).to(device)
		
		# Initialize model and log model info
		print(f'Building model...')
		model = CNN(len(classnames), len(X_train[0])).to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
		print(f'''Starting training:\nEpochs:        {epochs}\nBatch size:    {batch_size}\nLearning rate: {learning_rate}\nDevice:        {device.type}''')
		
		# Train the model!
		start_epoch = 0
		for epoch in range(start_epoch, epochs):
			train_loss, train_auroc, train_acc = train(epoch, trainloader, model, optimizer, device, len(classnames),weights)
			test_loss, test_auroc, test_acc = test(testloader, model, device, len(classnames), weights)
			val_loss, val_auroc, val_acc = test(valloader, model, device, len(classnames), weights)
			
			# Store each value in its corresponding list
			train_losses.append(train_loss)
			train_aurocs.append(train_auroc)
			train_accs.append(train_acc)
			test_losses.append(test_loss)
			test_aurocs.append(test_auroc)
			test_accs.append(test_acc)
			val_losses.append(val_loss)
			val_aurocs.append(val_auroc)
			val_accs.append(val_acc)
			
			# Log to console
			print(f'Train Loss: {train_loss}, Train AUROC: {train_auroc}, Train Accuracy: {train_acc}\nVal Loss: {val_loss}, Val AUROC: {val_auroc}, Val Accuracy: {val_acc}\nTest Loss: {test_loss}, Test AUROC: {test_auroc}, Test Accuracy: {test_acc}')
			
			if val_loss < best_loss:
				best_loss = val_loss
				early_stopping_counter = 0
				torch.save(model.state_dict(), f'{dir_checkpoint}/CNN_model.pt')  # Save the best model
				model_saved = True
			else:
				early_stopping_counter += 1
				
			# Check if early stopping criteria are met
			if early_stopping_counter >= patience:
				print(f"Early stopping triggered at epoch {epoch}")
				actual_epochs_ran = epoch
				break
		
		if model_saved:
			model = CNN(len(classnames), len(X_train[0])).to(device)
			model.load_state_dict(torch.load(f'{dir_checkpoint}/CNN_model.pt'))
		
		else:
			#from pathlib import Path
			#dir_checkpoint = '/'.join(self.modellingFile.split('/')[1:-1])
			#Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
			torch.save(model.state_dict(), f'{dir_checkpoint}/CNN_model.pt')
		
		# Plot training graphs
		eps = [i for i in range(actual_epochs_ran+1)]
		saveTrainingPlot(train_losses, eps, 'Train Loss', dir_checkpoint)
		saveTrainingPlot(train_aurocs, eps, 'Train AUROC', dir_checkpoint)
		saveTrainingPlot(train_accs, eps, 'Train Accuracy', dir_checkpoint)
		saveTrainingPlot(test_losses, eps, 'Test Loss', dir_checkpoint)
		saveTrainingPlot(test_aurocs, eps, 'Test AUROC', dir_checkpoint)
		saveTrainingPlot(test_accs, eps, 'Test Accuracy', dir_checkpoint)
		saveTrainingPlot(val_losses, eps, 'Val Loss', dir_checkpoint)
		saveTrainingPlot(val_aurocs, eps, 'Val AUROC', dir_checkpoint)
		saveTrainingPlot(val_accs, eps, 'Val Accuracy', dir_checkpoint)
		
		# Return train/test preds (and map to original labels!)
		with torch.no_grad(): 
			
			# Get test predictions
			x_dim = torch.unsqueeze(torch.Tensor(X_test), dim=1).float()
			test_logits = model(x_dim)
			test_probs = torch.softmax(test_logits, dim=1)
			test_preds_numeric = torch.Tensor([int(torch.argmax(prob)) for prob in test_probs]).to(device).type(torch.int)
			test_preds = label_encoder.inverse_transform(test_preds_numeric)
			
			# Get train predictions
			x_dim = torch.unsqueeze(torch.Tensor(X_train), dim=1).float()
			train_logits = model(x_dim)
			train_probs = torch.softmax(train_logits, dim=1)
			train_preds_numeric = torch.Tensor([int(torch.argmax(prob)) for prob in train_probs]).to(device).type(torch.int)
			train_preds = label_encoder.inverse_transform(train_preds_numeric)
			
			# Get train predictions
			x_dim = torch.unsqueeze(torch.Tensor(X_val), dim=1).float()
			val_logits = model(x_dim)
			val_probs = torch.softmax(val_logits, dim=1)
			val_preds_numeric = torch.Tensor([int(torch.argmax(prob)) for prob in val_probs]).to(device).type(torch.int)
			val_preds = label_encoder.inverse_transform(val_preds_numeric)

		return train_preds, test_preds, val_preds
	
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
		else:
			y_train_prob = softmax(y_train_prob, axis=1)

		y_test_preds = svm_model.predict(X_test)
		y_test_prob = svm_model.decision_function(X_test)
		if len(class_order)<=2:
			y_test_prob = expit(y_test_prob)
			y_test_prob = y_test_prob.reshape(-1,1)
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


		# if self.model_type == 'CNN':
		# 	classnames = list(set(y))
		# 	train_preds, test_preds, val_preds = self.runCNN(classnames,
		# 													X_train, X_val, X_test,
		# 													y_train, y_val, y_test)
		
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
		# acc_string = get_acc_string(X, y,
		# 							y_train, train_preds, 
		# 							y_test, test_preds, 
		# 							self.model_type, filename,
		# 							self.CNNHyperparameters,
		# 							y_val if self.model_type == 'CNN' else y_test, 
		# 							val_preds if self.model_type == 'CNN' else test_preds)
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



	def plot_latent_pca(self):
		peaks = self.df.iloc[0:, 4:].values
		labels =  self.df.iloc[0:, 0:2].values
		peaks = np.nan_to_num(peaks)
		pca = PCA(n_components=2)
		mm1 = MinMaxScaler()
		peaks_pca = pca.fit_transform( mm1.fit_transform( peaks ) )
		mm2 = MinMaxScaler()
		peaks_pca = mm2.fit_transform(peaks_pca)

		# plot PCA with colorcoding based on Slide and Class
		fig = plt.figure(figsize=(20,10))
		plot_titles = ['Slide distribution', 'Class distribution']
		for jj in range(2):
			ax = fig.add_subplot(1,2,jj+1)

			scatter_labels = labels[:,jj]
			legend_labels = np.unique(scatter_labels)
			n_colors = len(legend_labels)
			if n_colors<=10:
				class_colors = plt.cm.tab10(range(n_colors))
			else:
				class_colors = cm.get_cmap('jet_r')(np.linspace(0, 1, n_colors))

			for i in range(len(legend_labels)):
		
				ind = scatter_labels == legend_labels[i]
				xx = peaks_pca[ind]
				ax.scatter(xx[:,0],xx[:,1],
						color=class_colors[i],
						label=legend_labels[i], alpha=0.8)

			plt.legend()
			ax.set_xlabel('PC1')
			ax.set_ylabel('PC2')
			ax.set_ylim([0,1])
			ax.set_xlim([0,1])
			ax.set_title(plot_titles[jj])

		# save plot
		filename = self.modellingFile[:-4] + f'_PCAlatent.jpeg'
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


	# def LDA_plot(self, model, X_train, y_train):
	# 	import matplotlib.pyplot as plt
	# 	from matplotlib.colors import ListedColormap
	# 	import numpy as np
		
	# 	from matplotlib.axes._axes import _log as matplotlib_axes_logger

	# 	matplotlib_axes_logger.setLevel('ERROR')

	# 	data_plot = model.transform(X_train)

	# 	plt.figure()

	# 	for i, j in enumerate(np.unique(y_train)):
	# 			plt.scatter(data_plot[y_train == j, 0], data_plot[y_train == j, 1],
	# 									c = ListedColormap(('red', 'green', 'blue','black','cyan','magenta','yellow'))(i), label = j)

	# 	#add legend to plot
	# 	plt.legend(loc='best', shadow=False, scatterpoints=1)
		
	# 	# Add x and y axis
	# 	plt.xlabel('Linear discriminant 1')
	# 	plt.ylabel('Linear discriminant 2')

	# 	#display LDA plot
	# 	plt.savefig(self.modellingFile + 'scatter.jpeg')
	# 	self.saveFolder = self.modellingFile + 'scatter.jpeg'
  
	# uploads the modelling file and saves the file names into the class directory 
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
		print(self.savenameBase)

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
		filePaths = fileExplorer.getOpenFileName(None, "Open DESI text file", "", "Text Files (*.txt);;All Files (*)")
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
		self.peaks_3D = peaks.reshape((self.dim_y,self.dim_x,-1),order='C')
			
		# save path for pca image
		self.saveFolder = data_path
		self.slideName = slide_name[:-4]
		self.savenameBase = self.saveFolder+self.slideName
		print(self.savenameBase)


		# add each value
		self.selectedmz = []
		for i in range(len(mz)):
			self.selectedmz.append(mz[i])

		return True

	# def loadModellingTextFile(self, path):
	# 	[peaks, mz, dim_y, dim_x] = self.DESI_txt2numpy(path)
	# 	peaks_3D = peaks.reshape((dim_y,dim_x,-1),order='C')
	# 	name = path.split('/')[-1]
	# 	infostr = f'Slide successfully loaded \n'
	# 	infostr += f'File name:                 {name} \n'
	# 	infostr += f'Image dimensions:    {peaks_3D.shape[0]}x{peaks_3D.shape[1]} \n'
	# 	infostr += f'm/z per pixel:            {peaks_3D.shape[2]} \n'
	# 	infostr += f'Data size:                 {peaks_3D.shape}'
	# 	return infostr
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
		# convert to file to a numpy array and save as a matlab file
		[peaks, mz, dim_y, dim_x] = self.DESI_txt2numpy(name)
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
		self.peaks_3D = peaks.reshape((self.dim_y,self.dim_x,-1),order='C')
			
		# save path for pca image
		self.saveFolder = data_path
		self.slideName = slide_name[:-4]
		self.savenameBase = self.saveFolder+self.slideName
		print(self.savenameBase)


		# add each value
		self.selectedmz = []
		for i in range(len(mz)):
			self.selectedmz.append(mz[i])

		return True

	def getSelectedMz(self):
		return self.selectedmz
		
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
		# Import
		from scipy.signal import find_peaks

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
				mz_cell = new_mz_df[new_mz_df.columns[j]][i]
				if len(mz_cell)==0:
					pass
				elif len(mz_cell)==1:
					final_mz_df[final_mz_df.columns[j]][i] = mz_cell[0]
				else:
					ind = [np.abs(x-mz_old).argmin() for x in mz_cell]
					i_abundant = pmean[ind].argmax()
					final_mz_df[final_mz_df.columns[j]][i] = mz_cell[i_abundant]
								

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



	# def partial_pca_visulization(self, peaks, mask):
	# 	"""
	# 	create a PCA representation only using the pixels specified in segmentation mask
	# 	inputs:
	# 			peaks - 2D numpy array of DESI spectra with size [N_spectra x N_ions]
	# 			mask - 2D numpy array of segmentation mask [dim_y x dim_x] 
	# 	output:
	# 			pca_mask_image - 3D numpy array of the partial PCA representation [dim_y x dim_x x 3] 
	# 	example:
	# 			[peaks, mz, dim_y, dim_x] = DESI_txt2numpy(file)
	# 			roi_image = pltimage.imread(roi_file)
	# 			segmentation_mask = roi_image.sum(axis=2)>0
	# 			pca_mask_image = partial_pca_visulization(peaks, segmentation_mask)
	# 			plt.imshow(pca_mask_image)
	# 			plt.show()
	# 	author: @moon
	# 	"""
	# 	dim_y, dim_x = mask.shape
	# 	print(dim_y, dim_x)

	# 	peaks_image = peaks.reshape((dim_y,dim_x,-1),order='C')

	# 	X,Y = np.meshgrid( np.arange(dim_x), np.arange(dim_y) )
	# 	mask_x = X[mask][...,np.newaxis]
	# 	mask_y = Y[mask][...,np.newaxis]

	# 	mask_peaks = peaks_image[mask]
	# 	mask_tic = mask_peaks.sum(axis=1, keepdims=True)
	# 	mask_peaks = mask_peaks/mask_tic

	# 	pcaM = PCA(n_components=3)
	# 	pca_mask_data = pcaM.fit_transform(ion_minmax_normalize(mask_peaks))
	# 	pca_mask_data = ion_minmax_normalize(pca_mask_data)

	# 	pca_mask_image = np.zeros([dim_y,dim_x,3])
	# 	for i in range(len(mask_x)):
	# 			pca_mask_image[mask_y[i], mask_x[i], :] = pca_mask_data[i]
		
	# 	return pca_mask_image 


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



	# def roi_lda_display(self, mask, extend=False):
		
	# 	dim_y, dim_x = self.dim_y,self.dim_x

	# 	local_peaks_ind = mask>0
	# 	local_peaks_ind = local_peaks_ind.reshape((dim_y*dim_x,),order='C')

	# 	local_labels = mask.reshape((dim_y*dim_x,),order='C')
	# 	local_labels = local_labels[local_peaks_ind]

	# 	local_peaks = self.peaks_norm[local_peaks_ind]
	# 	local_pca = PCA(n_components=0.99)
	# 	local_scaler = MinMaxScaler()
	# 	local_peaks_pca = local_pca.fit_transform( local_scaler.fit_transform( local_peaks ) )
	# 	local_lda = LDA()
	# 	local_peaks_lda = local_lda.fit_transform(local_peaks_pca, local_labels)
	# 	# post_scaler = MinMaxScaler()
	# 	# post_scaler.fit(local_peaks_lda)

	# 	if extend:
	# 		# local_peaks_pca = local_pca.transform( local_scaler.transform( self.peaks_norm ) )
	# 		# local_peaks_lda = local_lda.transform(local_peaks_pca)
	# 		# local_peaks_lda = post_scaler.transform( local_peaks_lda )
	# 		# local_peaks_lda = np.clip(local_peaks_lda, 0,1)
	# 		# local_lda_image = local_peaks_lda.reshape((dim_y,dim_x,-1),order='C')
	# 		local_peaks_pca = local_pca.transform( local_scaler.transform( self.peaks_norm ) )
	# 		local_peaks_lda = local_lda.predict(local_peaks_pca)
	# 		local_lda_image = local_peaks_lda.reshape((1,dim_y,dim_x),order='C')
	# 	else:
	# 		# local_peaks_lda = post_scaler.transform( local_peaks_lda )
	# 		# local_peaks_lda = np.clip(local_peaks_lda, 0,1)
	# 		# # local_pca_image = self.peaks_pca.copy()
	# 		# local_lda_image = np.zeros((dim_y*dim_x,local_peaks_lda.shape[1]))
	# 		# local_lda_image[local_peaks_ind] = local_peaks_lda
	# 		# local_lda_image = local_lda_image.reshape((dim_y,dim_x,-1),order='C')
	# 		local_peaks_lda = local_lda.predict(local_peaks_pca)
	# 		local_lda_image = np.zeros((dim_y*dim_x,))
	# 		local_lda_image[local_peaks_ind] = local_peaks_lda
	# 		local_lda_image = local_lda_image.reshape((1,dim_y,dim_x),order='C')

	# 	local_lda_image = local_lda_image.astype("int")
	# 	local_lda_image = np.transpose(local_lda_image)

	# 	print(local_lda_image.shape, local_lda_image.dtype)
	# 	# labelMapNode = self.numpyArrayToSlicerLabelMap(local_lda_image, self.slideName+"_roi_lda")
	# 	ijkToRASMatrix = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])  # Your IJK to RAS matrix
	# 	labelMapNode = self.numpyArrayToSlicerLabelMap(local_lda_image, "MyLabelMap", ijkToRASMatrix)

	# 	segmentationNode = slicer.util.getNodesByClass('vtkMRMLSegmentationNode')[0]
	# 	customColorTableNode = self.createCustomColorTable(segmentationNode)
	# 	self.applyColormapToLabelMap(labelMapNode, customColorTableNode)

	# 	# Set the layout to 'Red Slice Only'
	# 	layoutManager = slicer.app.layoutManager()
	# 	layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)

	# 	# Get the red slice view's slice logic and set the label map node
	# 	redSliceWidget = layoutManager.sliceWidget('Red')
	# 	redSliceLogic = redSliceWidget.sliceLogic()
	# 	redSliceLogic.GetSliceCompositeNode().SetLabelVolumeID(labelMapNode.GetID())

	# 	# Ensure the display node of the label map node is visible
	# 	displayNode = labelMapNode.GetDisplayNode()
	# 	if displayNode:
	# 		displayNode.SetVisibility(1)


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

	# def numpyArrayToSlicerLabelMap(self, numpyArray, nodeName):
	# 	# print("numpyArray:", numpyArray, "nodeName:", nodeName)
	# 	# Convert numpy array to VTK array
	# 	vtkArray = numpy_support.numpy_to_vtk(num_array=numpyArray.flatten(), deep=True, array_type=vtk.VTK_INT)

	# 	# Create a vtkImageData object and set the VTK array as its scalars
	# 	imageData = vtk.vtkImageData()
	# 	imageData.SetDimensions(numpyArray.shape)
	# 	imageData.GetPointData().SetScalars(vtkArray)

	# 	# Create a new label map volume node
	# 	labelMapVolumeNode = slicer.vtkMRMLLabelMapVolumeNode()
	# 	labelMapVolumeNode.SetName(nodeName)
	# 	labelMapVolumeNode.SetAndObserveImageData(imageData)

	# 	# Add the label map volume node to the Slicer scene
	# 	slicer.mrmlScene.AddNode(labelMapVolumeNode)

	# 	return labelMapVolumeNode

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
		infostr = f'Slide successfully loaded \n'
		infostr += f'File name:\t \t {self.slideName} \n'
		infostr += f'Total number of spectra:\t {self.peaks_3D.shape[0]*self.peaks_3D.shape[1]} \n'
		infostr += f'Image dimensions:\t {self.peaks_3D.shape[0]} x {self.peaks_3D.shape[1]} pixels \n'
		infostr += f'Number of m/z per pixel:\t {self.peaks_3D.shape[2]} \n'
		return infostr
	
	def getREIMSInfo(self):
		infostr = f'Slide successfully loaded \n'
		infostr += f'File name:\t \t {self.slideName} \n'
		infostr += f'Total number of spectra:\t {self.dim_x} \n'
		infostr += f'Number of m/z per spectrum:\t {self.peaks_3D.shape[2]} \n'
		return infostr
	
	def datasetInfo(self, df):
		retstr = f'Number of slides:\t {len(set(df["Slide"]))}\n'
		retstr += f'Number of classes:\t {len(set(df["Class"]))}\n'
		retstr += f'Total number of spectra:\t {len(df)}\n'
		class_names,class_lens = np.unique(df["Class"], return_counts=1)
		for x,y in zip(class_names,class_lens):
			retstr += f'   {str(y)}\t in class\t {x} \n'
		return retstr
