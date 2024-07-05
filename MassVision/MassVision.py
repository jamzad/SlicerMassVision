
"""
MassVision

"""

from operator import truediv
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
# import os, unittest, logging, json
import logging
from MassVisionLib.Logic import * 


class MassVision(ScriptedLoadableModule):
	"""
	Uses ScriptedLoadableModule base class, available at:
	https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
	"""

	def __init__(self, parent):
		ScriptedLoadableModule.__init__(self, parent)

		self.parent.title = "MassVision"  
		self.parent.categories = ["Spectral Imaging"]
		self.parent.dependencies = []
		self.parent.contributors = ["Med-i Lab, Queen's University (Amoon Jamzad, Jade Warren, Ayesha Syeda)"] 
		self.parent.helpText = """
		MassVision is a software solution developed in 3D Slicer platform for end-to-end analysis of Mass Spectrometry Imaging (MSI) data, particularly Desorption ElectroSpray Ionization (DESI) modality. 
		
		The functionalities include data exploration via various targeted and non-targeted visualization, co-localization to spatial labels (histopathology annotations), dataset generation with spatial- and spectral-guidance, multi-slide data aggregation via feature alignment, denoising via spatial aggregation, machine learning model training, and whole-slide model deployment.
		
		Please cite the following publication: TBA MassVision"""
		self.parent.acknowledgementText = """

		"""

	
	def setupUi(self, parent):
		# Call the superclass implementation
		ScriptedLoadableModule.setupUi(self, parent)

		# Get a reference to your QTabWidget
		tabWidget = self.ui.tabWidget


class MassVisionTest(ScriptedLoadableModuleTest):
	"""
	This is the test case for your scripted module.
	Uses ScriptedLoadableModuleTest base class, available at:
	https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
	"""

	def setUp(self):
		""" Do whatever is needed to reset the state - typically a scene clear will be enough.
		"""
		slicer.mrmlScene.Clear()

	def runTest(self):
		"""Run as few or as many tests as needed here.
		"""
		self.setUp()
		self.test_MassVision1()

	def test_MassVision1(self):
		""" Ideally you should have several levels of tests.  At the lowest level
		tests should exercise the functionality of the logic with different inputs
		(both valid and invalid).  At higher levels your tests should emulate the
		way the user would interact with your code and confirm that it still works
		the way you intended.
		One of the most important features of the tests is that it should alert other
		developers when their changes will have an impact on the behavior of your
		module.  For example, if a developer removes a feature that you depend on,
		your test should break so they know that the feature is needed.
		"""

		self.delayDisplay("Starting the test")

		logic = MassVisionLogic()

		self.delayDisplay('Test passed')

class MassVisionWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
	"""
 	Uses ScriptedLoadableModuleWidget base class, available at:
	https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
	"""

	def __init__(self, parent=None):
		"""
		Called when the user opens the module the first time and the widget is initialized.
		"""
		ScriptedLoadableModuleWidget.__init__(self, parent)
		VTKObservationMixin.__init__(self)  # needed for parameter node observation
		self.logic = None
		self._parameterNode = None
		self._updatingGUIFromParameterNode = False
		self.parameterSetNode = None
		self.REIMS = 0

	def setup(self):
		"""
		Called when the user opens the module the first time and the widget is initialized.
		"""
		ScriptedLoadableModuleWidget.setup(self)

		# Load widget from .ui file (created by Qt Designer).
		uiWidget = slicer.util.loadUI(self.resourcePath('UI/MassVision.ui'))
		self.layout.addWidget(uiWidget)
		self.ui = slicer.util.childWidgetVariables(uiWidget)

		# Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
		# "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
		# "setMRMLScene(vtkMRMLScene*)" slot.
		#self.ui.subjectHierarchy.setMRMLScene(slicer.mrmlScene)

		# Create logic class. Logic implements all computations that should be possible to run
		# in batch mode, without a graphical user interface.
		self.cases_config = {}
		self.logic = MassVisionLogic()

		# set the first tab as the default loading tab
		self.ui.tabWidget.setCurrentIndex(0)
		self.ui.tabWidget.currentChanged.connect(self.onTabChange)
		slicer.util.moduleSelector().connect('moduleSelected(QString)', self.onModuleChange)

		# Connections
		if self.REIMS==0:
			self.ui.label_REIMS.hide()
			self.ui.REIMSFileSelect.hide()
			self.ui.filenameREIMSBrowser.hide()
			self.ui.REIMSFileLoad.hide()
			self.ui.REIMSInformation.hide()

		# import push botton mechanism
		self.ui.textFileLoad.hide()
		self.ui.loadHisto.hide()
		self.ui.csvLoad.hide()
		self.ui.modellingFile.hide()
		self.ui.deployImport.hide()
		self.ui.deployModelImport.hide()
		
		# Data Import
		self.ui.clearReloadPush.connect("clicked(bool)", self.onClearReload)
		self.ui.loadScenePush.connect("clicked(bool)", self.onLoadScene)

		self.ui.textFileSelect.connect("clicked(bool)", self.onTextFileSelect)
		self.ui.textFileLoad.connect("clicked(bool)", self.onTextFileLoad)

		self.ui.histoFileSelect.connect("clicked(bool)",self.onHistoSelect)
		self.ui.loadHisto.connect("clicked(bool)",self.onloadHisto)

		self.ui.REIMSFileSelect.connect("clicked(bool)",self.onREIMSSelect)
		self.ui.REIMSFileLoad.connect("clicked(bool)",self.onREIMSLoad)


		# Visualization
		self.ui.spectrumPlot.connect("clicked(bool)", self.onSpectrumPlot)
		self.ui.singleIonHeatmapList.addItem('Inferno')
		self.ui.singleIonHeatmapList.addItem('DivergingBlueRed')
		self.ui.singleIonHeatmapList.addItem('PET-Rainbow2')
		self.ui.singleIonHeatmapList.addItem('Cividis')
		self.ui.singleIonHeatmapList.addItem('ColdToHotRainbow')
	
		self.ui.singleIonButton.connect("clicked(bool)", self.selectedSingleIon)
		self.ui.multiIonButton.connect("clicked(bool)", self.selectedMultiIon)
		self.ui.PCA_button.connect("clicked(bool)", self.onPCAButton)
		self.ui.partialPCA.connect("clicked(bool)", self.onPartialPCAButton)
		self.dataInfo = ''
		

		# Dataset generation
		self.ui.gotoRegistration.connect("clicked(bool)", self.landmark)
		self.ui.segmentEditor.connect("clicked(bool)", self.showSegmentEditor)
		self.ui.roiContrast.connect("clicked(bool)", self.onROIContrast)
		self.ui.roiContrastLDA.connect("clicked(bool)", self.onROIContrastLDA)
		self.ui.segmentVisibility.connect("clicked(bool)", self.onSegmentVisibility)
		self.ui.createCSVbutton.connect("clicked(bool)",self.onCSVconnect)
		self.ui.saveScenePush.connect("clicked(bool)",self.onSaveScene)


		# Multi-slide alignment
		self.ui.pushButton_11.connect("clicked(bool)", self.onMerge)
		self.ui.textFileLoad_2.connect("clicked(bool)", self.onCSVmerge)
		self.files = set()
  

		# Dataset post-processing
		self.ui.csvSelect.connect("clicked(bool)", self.onCsvSelect)
		self.ui.csvLoad.connect("clicked(bool)", self.onCsvLoad)
		self.ui.normalizeCheckbox.connect("clicked(bool)", self.onNormalizationState)

		self.ui.refNorm.toggled.connect(self.onIonNorm)
		self.ui.normalizeTICoption.toggled.connect(self.onIonNorm)
		
		self.ui.spectrumFiltercheckBox.connect("clicked(bool)", self.onFilterState)
		self.ui.pixelaggcheckBox.connect("clicked(bool)", self.onAggState)
		self.ui.applyProcessingButton.connect("clicked(bool)", self.onApplyProcessing)	
  

		# Model Training
		self.ui.selectCSV.connect("clicked(bool)", self.onSelectModelData)
		self.ui.modellingFile.connect("clicked(bool)", self.onModellingLoad)
		self.ui.distributionPCA.connect("clicked(bool)", self.onPlotDIstribution)

		self.ui.randomSplit.connect("clicked(bool)", self.onRandomSplit) # random
		self.ui.customSplit.connect("clicked(bool)", self.onCustomSplit)  # custom
		self.ui.allTrain.connect("clicked(bool)", self.onAllTrainSplit)  # all train

		self.ui.trainModel.connect("clicked(bool)", self.onModelTrain)
		self.onAllTrainSplit()
		

		# Results
		self.model_results = ''


		# Deployment
		self.ui.deploySelect.connect("clicked(bool)", self.onDeploySelect)
		self.ui.deployImport.connect("clicked(bool)", self.onDeployLoad)
		self.ui.deployModelSel.connect("clicked(bool)", self.onDeployModelSel)
		self.ui.deployModelImport.connect("clicked(bool)", self.onDeployModelLoad)
		self.ui.deployNormcheck.connect("clicked(bool)", self.onDeployNormCheck)

		self.ui.depRadioTIC.toggled.connect(self.onDepNormRadioToggle)
		self.ui.depRadioIon.toggled.connect(self.onDepNormRadioToggle)

		self.ui.deployAGGcheck.connect("clicked(bool)", self.onDeployAggCheck)

		self.ui.depMaskcheck.connect("clicked(bool)", self.onDepMaskcheck)
		self.ui.depGoVisButton.connect("clicked(bool)", self.onDepGoVis)
		self.ui.depGoSegEdButton.connect("clicked(bool)", self.onDepGoSeg)

		self.ui.deployRun.connect("clicked(bool)", self.onApplyDeployment)	


		# Make sure parameter node is initialized (needed for module reload)
		self.initializeParameterNode()


	### Data Import Tab

	def onClearReload(self):
		confirmation_dialog = qt.QMessageBox()
		confirmation_dialog.setIcon(qt.QMessageBox.Question)
		confirmation_dialog.setText("Are you sure you want to delete all files and clear the scene?")
		confirmation_dialog.setWindowTitle("Confirmation")
		confirmation_dialog.setStandardButtons(qt.QMessageBox.Yes | qt.QMessageBox.No)

		# Show the confirmation dialog and get the user's choice
		choice = confirmation_dialog.exec_()

		# Check the user's choice
		if choice == qt.QMessageBox.Yes:
			slicer.mrmlScene.Clear()
			slicer.util.reloadScriptedModule('MassVision')
			print('MODULE RELOADED')



	def onLoadScene(self):
		fileExplorer = qt.QFileDialog()
		mrbFilename = fileExplorer.getOpenFileName(None, "Load Slicer Project", "", "MSI Project Files (*.mrb);;All Files (*)")
		slicer.util.loadScene(mrbFilename)

	def onTextFileSelect(self):
		file_info = self.logic.textFileSelect()
		if file_info!=('',''):
			# self.ui.filenameTextBrowser.setText(f'{file_info[0]}{file_info[1]}')
			self.ui.ImportlineEdit.setText(f'{file_info[0]}{file_info[1]}')
			self.ui.ImportlineEdit.setToolTip(f'{file_info[0]}{file_info[1]}')
			self.onTextFileLoad()

	def onTextFileLoad(self):
		# on the text file load runs the text file load and shows the confirmation button

		# file_load = self.logic.textFileLoad(self.ui.filenameTextBrowser.toPlainText())
		file_load = self.logic.textFileLoad(self.ui.ImportlineEdit.text)
		if file_load:
			tic_normalized = self.logic.normalize()
			info = self.logic.getDataInformation() if tic_normalized else 'Error in TIC Normalization'
		else:
			info = 'Error in File Load. Please check the data and try again.'

		self.dataInfo = info
		self.ui.dataInformation.setText(info)
		self.logic.heatmap_display()
		self.populateMzLists()
  
	def onHistoSelect(self):
		# histoPath = self.logic.fileSelect()
		histoPath = self.logic.HistofileSelect()
		if histoPath:
			self.ui.HistoLineEdit.setText(histoPath)
			self.ui.HistoLineEdit.setToolTip(histoPath)
			self.onloadHisto()
			

	def onloadHisto(self):
		# when load histopathology is selected runs load histopathology
		self.logic.loadHistopathology(self.ui.HistoLineEdit.text)


	def onREIMSSelect(self):
		file_info = self.logic.REIMSSelect()
		if file_info:
			self.ui.filenameREIMSBrowser.setText(f'{file_info[0]}{file_info[1]}')

	def onREIMSLoad(self):
		# on the text file load runs the text file load and shows the confirmation button

		file_load = self.logic.REIMSLoad(self.ui.filenameREIMSBrowser.toPlainText())
		if file_load:
			tic_normalized = self.logic.normalize()
			info = self.logic.getREIMSInfo() if tic_normalized else 'Error in TIC Normalization'
		else:
			info = 'Error in File Load. Please check the data and try again.'

		self.dataInfo = info
		self.ui.REIMSInformation.setText(info)
		self.logic.heatmap_display()
		self.populateMzLists()


	### Visualization tab
	def onSpectrumPlot(self):
		self.logic.spectrum_plot()
		return True
	
	def onPartialPCAButton(self):
		# get all ROIs
		all_rois = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsROINode")
		# select the last ROI created
		roi = all_rois.GetItemAsObject(all_rois.GetNumberOfItems()-1)
		# get roi bounds
		bounds = np.zeros((6,1))
		roi.GetBounds(bounds)
		# error checking to ensure it is within image bounds
		max_x = min(self.logic.dim_x, np.ceil(abs(bounds[0])))
		min_x = 0 if bounds[1] > 0 else np.ceil(-bounds[1])
		max_y = min(self.logic.dim_y, np.ceil(abs(bounds[2])))
		min_y = 0 if bounds[3] > 0 else np.ceil(-bounds[3])

		print(f'partial PCA region: ({min_y}, {max_y}) ({min_x}, {max_x})')
		# logic processes pca in the ROI
		self.logic.partial_pca_display((int(min_y), int(min_x)), (int(max_y), int(max_x)), 
								 extend=self.ui.pcaExtendCheckbox.isChecked())

	def onROIContrast(self):
		# get all ROIs
		segmentationNode = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLSegmentationNode')
		segments = segmentationNode.GetSegmentation()
		n_segments = segments.GetNumberOfSegments()
		label_mask = np.zeros((self.logic.dim_y, self.logic.dim_x), dtype='bool')

		for i in range(n_segments):
			segment_id = segments.GetNthSegmentID(i)
			try:
				a = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segment_id)
				a = a*(i+1)
			except:
				referenceVolume = [x for x in slicer.mrmlScene.GetNodesByClass('vtkMRMLScalarVolumeNode') if 'tic' in x.GetName()][0]
				a = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segment_id, referenceVolume)
				a = a*(i+1)
			label_mask = label_mask + a
		
		# logic processes pca in the ROI
		self.logic.roi_pca_display(label_mask, extend=self.ui.roiCintrastExtend.isChecked())


	def onROIContrastLDA(self):
		# get all ROIs
		segmentationNode = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLSegmentationNode')
		segments = segmentationNode.GetSegmentation()
		n_segments = segments.GetNumberOfSegments()
		label_mask = np.zeros((self.logic.dim_y, self.logic.dim_x), dtype='bool')

		for i in range(n_segments):
			segment_id = segments.GetNthSegmentID(i)
			try:
				a = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segment_id)
				a = a*(i+1)
			except:
				referenceVolume = [x for x in slicer.mrmlScene.GetNodesByClass('vtkMRMLScalarVolumeNode') if 'tic' in x.GetName()][0]
				a = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segment_id, referenceVolume)
				a = a*(i+1)
			label_mask = label_mask + a
		
		# logic processes pca in the ROI
		self.logic.roi_lda_display(label_mask, extend=self.ui.roiCintrastExtend.isChecked())


	def onSegmentVisibility(self):
		lm = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLSegmentationNode')
		vis = lm.GetDisplayVisibility()
		if vis:
			lm.SetDisplayVisibility(0)
		else:
			lm.SetDisplayVisibility(1)

	def onPCAButton(self):
		# displays the pca image
		self.logic.pca_display()
  
	def populateMzLists(self):
		# adds the m/z values to m/z lists for the color channels
		self.ui.mzlist.clear()
		self.ui.mzlist_2.clear()
		self.ui.mzlist_3.clear()
		self.ui.mzlist_4.clear()
		self.ui.mzlist_5.clear()
		self.ui.mzlist_6.clear()
		self.ui.mzlist_7.clear()
		self.ui.singleIonMzList.clear()

		self.ui.mzlist.addItem("None")
		self.ui.mzlist_2.addItem("None")
		self.ui.mzlist_3.addItem("None")
		self.ui.mzlist_4.addItem("None")
		self.ui.mzlist_5.addItem("None")
		self.ui.mzlist_6.addItem("None")
		self.ui.mzlist_7.addItem("None")
		self.ui.singleIonMzList.addItem("None")
		
		all_mz = self.logic.getSelectedMz()
		for mz in all_mz:
			self.ui.mzlist.addItem(mz)
			self.ui.mzlist_2.addItem(mz)
			self.ui.mzlist_3.addItem(mz)
			self.ui.mzlist_4.addItem(mz)
			self.ui.mzlist_5.addItem(mz)
			self.ui.mzlist_6.addItem(mz)
			self.ui.mzlist_7.addItem(mz)
			self.ui.singleIonMzList.addItem(mz)

		
	def selectedMultiIon(self):
		# runs the valudation for all of the color channels
		self.logic.multiIonVisualization([["white",self.ui.mzlist.currentText], 
									["red",self.ui.mzlist_2.currentText],
									["green",self.ui.mzlist_3.currentText],
									["blue",self.ui.mzlist_4.currentText],
									["yellow",self.ui.mzlist_7.currentText], 
									["magenta",self.ui.mzlist_5.currentText],
									["cyan",self.ui.mzlist_6.currentText]])
		
	def selectedSingleIon(self):
		# runs the valudation for all of the color channels
		self.logic.singleIonVisualization(float(self.ui.singleIonMzList.currentText), 
									self.ui.singleIonHeatmapList.currentText)

	### Dataset generation

	def landmark(self):
		# swtiches module to landmark registration
		pluginHandlerSingleton = slicer.qSlicerSubjectHierarchyPluginHandler.instance()
		pluginHandlerSingleton.pluginByName('Default').switchToModule("LandmarkRegistration")

	def showSegmentEditor(self):
		segVol1 = self.ui.segVollist1.currentText
		segVol2 = self.ui.segVollist2.currentText

		if segVol1!='None':
		
			slicer.util.selectModule("SegmentEditor")

			# set master volume and geometry
			segSelect1Node = slicer.util.getNode( segVol1 )
			sourceVolumeNode = segSelect1Node
			segmentEditorNode = slicer.util.getNodesByClass('vtkMRMLSegmentEditorNode')[0]
			segmentationNode = slicer.util.getNodesByClass('vtkMRMLSegmentationNode')[0]
			segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(sourceVolumeNode)
			segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
			segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
			segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
			segmentEditorWidget.setSegmentationNode(segmentationNode)
			segmentEditorWidget.setSourceVolumeNode(sourceVolumeNode)

			# set the display view and link views
			RedCompNode = slicer.util.getNode("vtkMRMLSliceCompositeNodeRed")
			RedCompNode.SetBackgroundVolumeID(segSelect1Node.GetID())
			RedNode = slicer.util.getNode("vtkMRMLSliceNodeRed")
			RedNode.SetOrientation("Axial")

			if segVol2=='None':
				slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)
				slicer.util.resetSliceViews()

			else:
				segSelect2Node = slicer.util.getNode( segVol2 )
				YellowCompNode = slicer.util.getNode("vtkMRMLSliceCompositeNodeYellow")
				YellowCompNode.SetBackgroundVolumeID(segSelect2Node.GetID())
				YellowNode = slicer.util.getNode("vtkMRMLSliceNodeYellow")
				YellowNode.SetOrientation("Axial")
				RedCompNode.SetLinkedControl(True)
				YellowCompNode.SetLinkedControl(True)
				slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutSideBySideView)
				slicer.util.resetSliceViews()

  
	def onCSVconnect(self):
		fileExplorer = qt.QFileDialog()
		# defaultSave = self.ui.filenameTextBrowser.toPlainText()[:-4]+'_dataset'
		defaultSave = self.ui.ImportlineEdit.text[:-4]+'_dataset'
		
		savepath = fileExplorer.getSaveFileName(None, "Save aligned dataset", defaultSave, "CSV Files (*.csv);;All Files (*)")
		
		retstr = self.logic.csvGeneration(savepath)
		self.ui.csvcreateTextBrowser.setText(retstr)
		self.logic.segmentationSave(savepath)
	
	def onSaveScene(self):
		print('saving the project...')
		fileExplorer = qt.QFileDialog()
		# defaultSave = self.ui.filenameTextBrowser.toPlainText()[:-4]+'_project'
		defaultSave = self.ui.ImportlineEdit.text[:-4]+'_project'
		savepath = fileExplorer.getSaveFileName(None, "Save Project", defaultSave, "MSI Project Files (*.mrb);;All Files (*)")
		print(savepath)
		slicer.util.saveScene(savepath)


	def onTabChange(self, index):
		print('selected tab:',index, self.ui.tabWidget.tabText(index))
		if index==7:
			self.updateDepVisList()
			self.updateDepSegList()
		elif index==2:
			self.updateVolumeList()
	
	def onModuleChange(self):
		self.updateDepVisList()
		self.updateDepSegList()
		self.updateVolumeList()

	def updateVolumeList(self):
		self.ui.segVollist1.clear()
		self.ui.segVollist1.addItem('None')
		self.ui.segVollist2.clear()
		self.ui.segVollist2.addItem('None')
		volumeNodes = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')
		for volumeNode in volumeNodes:
			self.ui.segVollist1.addItem(volumeNode.GetName())
			self.ui.segVollist2.addItem(volumeNode.GetName())

	def updateDepVisList(self):
		self.ui.depVisListCombo.clear()
		volumeNodes = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')
		for volumeNode in volumeNodes:
			self.ui.depVisListCombo.addItem(volumeNode.GetName())

	def updateDepSegList(self):
		segmentationNode = slicer.util.getNodesByClass('vtkMRMLSegmentationNode')
		if len(segmentationNode)>0:
			segmentationNode = segmentationNode[0]
			segmentation = segmentationNode.GetSegmentation()
			segIDs = segmentation.GetSegmentIDs()
			segNames = [segmentation.GetSegment(segID).GetName() for segID in segIDs]

			self.ui.depSegListCombo.clear()
			for segName in segNames:
				self.ui.depSegListCombo.addItem(segName)


	### Multi-slide alignment tab
 
	def onMerge(self):
		# Merge csv files added by the user
		fileExplorer = qt.QFileDialog()
		defaultSave = list(self.files)[-1][:-4]+'_aligned'
		savepath = fileExplorer.getSaveFileName(None, "Save aligned dataset", defaultSave, "CSV Files (*.csv);;All Files (*)")
		print('save path:',savepath)
		retstr = self.logic.batch_peak_alignment(list(self.files), savepath)
		self.ui.alignmentTextBrowser.setText(retstr)
	
	def onCSVmerge(self):
		# gets the list of files they are trying to merge and shows them in the viewer
		fileExplorer = qt.QFileDialog()
		filePaths = fileExplorer.getOpenFileNames(None, "Open CSV datasets", "", "CSV Files (*.csv);;All Files (*)")
		for filePath in filePaths:
			if filePath not in self.files:
				i = len(self.files)
				self.files.add(filePath)
				self.ui.filesTable.setRowCount(len(self.files))
				delete_btn = qt.QPushButton('Delete')
				delete_btn.clicked.connect(self.createDeleteButtonClickedHandler(filePath))
				self.ui.filesTable.setItem(i, 0, qt.QTableWidgetItem(filePath))
				self.ui.filesTable.setCellWidget(i, 1, delete_btn)
	
		# Wrapper functions to let us pass the filenames to the button press handler
	def createDeleteButtonClickedHandler(self, path):
		def deleteButtonPressed(checked):
			self.handleDeleteButtonClicked(checked, path)
		return deleteButtonPressed

	def handleDeleteButtonClicked(self, checked, path=None):
		self.files.remove(path)
		r, numRows = 0, self.ui.filesTable.rowCount
		while r < numRows:
			if self.ui.filesTable.item(r, 0).text() == path:
				self.ui.filesTable.removeRow(r)
				self.ui.filesTable.setRowCount(len(self.files))
				numRows -= 1
			r += 1
   
	### Dataset post-processing
	def onCsvSelect(self):
		fileExplorer = qt.QFileDialog()
		csvFilename = fileExplorer.getOpenFileName(None, "Open CSV dataset", "", "CSV Files (*.csv);;All Files (*)")
		self.csvForProcess = csvFilename
		if csvFilename:
			self.ui.PostProcessinglineEdit.setText(csvFilename)
			self.ui.PostProcessinglineEdit.setToolTip(csvFilename)
			self.onCsvLoad()

	def onCsvLoad(self):
		csv_info = self.logic.CsvLoad(self.ui.PostProcessinglineEdit.text)
		if csv_info:
			self.ui.csvInfo.setText(csv_info)
			self.ui.refIoncomboBox.clear()
			all_mz = self.logic.getCsvMzList()
			for mz in all_mz:
				self.ui.refIoncomboBox.addItem(mz)
	
	def onNormalizationState(self):
		if self.ui.normalizeCheckbox.isChecked():
			self.ui.normalizeTICoption.setEnabled(True)
			self.ui.refNorm.setEnabled(True)
			self.onIonNorm()
		else:
			self.ui.normalizeTICoption.setEnabled(False)
			self.ui.refNorm.setEnabled(False)
			self.ui.refIoncomboBox.setEnabled(False)

	def onIonNorm(self):
		if self.ui.refNorm.isChecked():
			self.ui.refIoncomboBox.setEnabled(True)
		else:
			self.ui.refIoncomboBox.setEnabled(False)


	def onFilterState(self):
		if self.ui.spectrumFiltercheckBox.isChecked():
			State = True
		else:
			State = False
		self.ui.spectrumUpperband.setEnabled(State)
		self.ui.spectrumlowerBand.setEnabled(State)
		self.ui.lowLabel.setEnabled(State)
		self.ui.upLabel.setEnabled(State)

	def onAggState(self):
		if self.ui.pixelaggcheckBox.isChecked():
			State = True
		else:
			State = False
		self.ui.patchWidth.setEnabled(State)
		self.ui.patchStride.setEnabled(State)
		self.ui.partialPatch.setEnabled(State)
		self.ui.aggMode.setEnabled(State)
		self.ui.wLabel.setEnabled(State)
		self.ui.sLabel.setEnabled(State)
		self.ui.pLabel.setEnabled(State)
		self.ui.mLabel.setEnabled(State)
	
	def onApplyProcessing(self):
		# get normalize parameters
		if self.ui.normalizeCheckbox.isChecked():
			if self.ui.normalizeTICoption.isChecked():
				spec_normalization = 'tic'
			else:
				spec_normalization = self.ui.refIoncomboBox.currentText
		else:
			spec_normalization = None
	
		# get spectrum filtering
		if self.ui.spectrumFiltercheckBox.isChecked():
			subband_selection = (int(self.ui.spectrumlowerBand.text),int(self.ui.spectrumUpperband.text))
		else:
			subband_selection = None
   
		# get pixel aggregation
		if self.ui.pixelaggcheckBox.isChecked():
			pixel_aggregation = (int(self.ui.patchWidth.text), int(self.ui.patchStride.text), self.ui.aggMode.currentText, int(self.ui.partialPatch.text))
		else:
			pixel_aggregation = None

		fileExplorer = qt.QFileDialog()
		defaultSave = self.csvForProcess[:-4]+'_processed.csv'
		savepath = fileExplorer.getSaveFileName(None, "Save processed dataset", defaultSave, "CSV Files (*.csv);;All Files (*)")
		print(savepath)


		processed_csv_info = self.logic.dataset_post_processing(spec_normalization, subband_selection, pixel_aggregation, savepath)

		retstr = 'Dataset successfully processed! \n'
		retstr += f'Processed dataset:\t {savepath} \n'
		retstr += processed_csv_info
		self.ui.postCsvinfo.setText(retstr)


	### Model training tab

	def onPlotDIstribution(self):
		self.logic.plot_latent_pca()

	def onSelectModelData(self):
		fileExplorer = qt.QFileDialog()
		csvFilename = fileExplorer.getOpenFileName(None, "Open CSV dataset", "", "CSV Files (*.csv);;All Files (*)")
		if csvFilename:
			self.ui.ModellineEdit.setText(csvFilename)
			self.ui.ModellineEdit.setToolTip(csvFilename)
			self.onModellingLoad()

	def onModellingLoad(self):
		# loads the file for modelling and does confirmation pop up 
		dataset_info = self.logic.modellingFileLoad(self.ui.ModellineEdit.text)
		if not dataset_info:
			dataset_info = 'Error in file load. Please check the console error and try again.'
		else:
			self.ui.dataInformation_2.setText(dataset_info)

		# from functools import partial
		self.cases_config = {}
		names_list = self.logic.custom_split()
		self.ui.namesTable.setRowCount(len(names_list))
		self.ui.namesTable.setColumnCount(4)
		self.ui.namesTable.setHorizontalHeaderLabels(['Name', 'Train', 'Test', 'Validation'])

		for i, name in enumerate(names_list):
			train_btn, test_btn, val_btn = qt.QCheckBox(), qt.QCheckBox(), qt.QCheckBox()
			train_btn.clicked.connect(self.createRadioButtonClickedHandler(name, train_btn))
			test_btn.clicked.connect(self.createRadioButtonClickedHandler(name, test_btn))
			val_btn.clicked.connect(self.createRadioButtonClickedHandler(name, val_btn))
			self.cases_config[name] = [train_btn, test_btn, val_btn]
			self.ui.namesTable.setItem(i, 0, qt.QTableWidgetItem(name))
			self.ui.namesTable.setCellWidget(i, 1, train_btn)
			self.ui.namesTable.setCellWidget(i, 2, test_btn)
			self.ui.namesTable.setCellWidget(i, 3, val_btn)
			train_btn.setChecked(True)

		self.ui.namesTable.setColumnHidden(3, True)

		if self.ui.randomSplit.isChecked():
			self.logic.set_split('random')
		elif self.ui.allTrain.isChecked():
			self.logic.set_split('all_train')
		

	
	def updateCustomConfig(self):
		"""
  		Let logic keep track of what cases are train/test.
		"""
		train_cases, test_cases, val_cases = set(), set(), set()
		for case, buttons in self.cases_config.items():
			train_btn, test_btn, val_btn = buttons
			if train_btn.isChecked(): train_cases.add(case)
			if test_btn.isChecked(): test_cases.add(case)
			if val_btn.isChecked(): val_cases.add(case)
		self.logic.update_test_cases(test_cases)
		self.logic.update_train_cases(train_cases)
		self.logic.update_val_cases(val_cases)
	

	def onRandomSplit(self):
		"""
  		Let logic keep track of what split is being used.
		"""
		self.ui.namesTable.hide()
		self.ui.namesTableLabel.hide()
		self.logic.set_split('random')

	def onAllTrainSplit(self):
		"""
  		Let logic keep track of what split is being used.
		"""
		self.ui.namesTable.hide()
		self.ui.namesTableLabel.hide()
		self.logic.set_split('all_train')

	def onCustomSplit(self):
		"""
		Populate namesTable.
  		
		For each case, adds a row with the case name and checkboxes
		so the user can select if they want it to be in train/test.
		"""
		self.ui.namesTable.show()
		self.ui.namesTableLabel.show()
		self.logic.set_split('custom')
			
	# Wrapper functions to let us pass the buttons to the function,
	# making the checkboxes exclusive. 

	def createRadioButtonClickedHandler(self, name, checkbox):
		def radioButtonClicked(checked):
			self.handleRadioButtonClicked(checked, name, checkbox)
		return radioButtonClicked

	def handleRadioButtonClicked(self, checked, name=None, checkbox=None):
		train_btn, test_btn, val_btn = self.cases_config[name]
		if (train_btn.isChecked() or val_btn.isChecked()) and checkbox == test_btn and checked:
			train_btn.setChecked(False)
			val_btn.setChecked(False)
			test_btn.setChecked(True)
		if (test_btn.isChecked() or val_btn.isChecked()) and checkbox == train_btn and checked:
			test_btn.setChecked(False)
			val_btn.setChecked(False)
			train_btn.setChecked(True)
		if (test_btn.isChecked() or train_btn.isChecked()) and checkbox == val_btn and checked:
			test_btn.setChecked(False)
			train_btn.setChecked(False)
			val_btn.setChecked(True)
   
	def onModelTrain(self):
		"""
		Train model according to user specifications.
		"""
		savepath = None
		if self.ui.saveModelcheckBox.isChecked():
			fileExplorer = qt.QFileDialog()
			defaultSave = self.logic.modellingFile[:-4]+"_model.pkl"
			savepath = fileExplorer.getSaveFileName(None, "Save model pipeline", defaultSave, "Pickle Files (*.pkl);;All Files (*)")
			print(savepath)

		self.updateCustomConfig()
		
		self.logic.model_type = self.ui.ModelSelectCombobox.currentText
		self.logic.train_balancing = self.ui.BalanceComBox.currentText
		
		accuracystring = self.logic.runModel(savepath)
		if not accuracystring:
			self.ui.textBrowser.setText('An error occured. Please check the console for details.')
		else:
			self.ui.textBrowser.setText(accuracystring)
			self.model_results = accuracystring
			self.ui.tabWidget.setCurrentIndex(6)

  
	### Model deployment tab
	def onDeploySelect(self):
		file_loc = self.logic.textFileSelect()
		if file_loc!=('',''):
			self.ui.deployLoclineEdit.setText(f'{file_loc[0]}{file_loc[1]}')
			self.ui.deployLoclineEdit.setToolTip(f'{file_loc[0]}{file_loc[1]}')
			self.onDeployLoad()
   
	def onDeployLoad(self):
		# on the text file load runs the text file load and shows the confirmation button
		self.logic.textFileLoad(self.ui.deployLoclineEdit.text)
		info = self.logic.getDataInformation()
		self.ui.deployInfo.setText(info)
		## make the visualization options available for this slide
		self.logic.normalize()
		self.logic.heatmap_display()
		self.populateMzLists()
		self.updateDepVisList()
  
	def onDeployModelSel(self):
		fileExplorer = qt.QFileDialog()
		pklFilename = fileExplorer.getOpenFileName(None, "Open model pipeline", "", "Pickle Files (*.pkl);;All Files (*)")
		self.pklFilename = pklFilename
		if pklFilename:
			self.ui.deployModellineEdit.setText(pklFilename)
			self.ui.deployModellineEdit.setToolTip(pklFilename)
			self.onDeployModelLoad()

	def onDeployModelLoad(self):
		pipelineInfo = self.logic.loadPipeline(self.ui.deployModellineEdit.text)
		self.ui.deployModelInfo.setText(pipelineInfo)
		self.ui.depComboboxIon.clear()
		for mz in self.logic.DmzRef:
			self.ui.depComboboxIon.addItem(mz)

	def onDeployNormCheck(self):
		if self.ui.deployNormcheck.isChecked():
			self.ui.depRadioTIC.setEnabled(True)
			self.ui.depRadioIon.setEnabled(True)
			self.onDepNormRadioToggle()
		else:
			self.ui.depRadioTIC.setEnabled(False)
			self.ui.depRadioIon.setEnabled(False)
			self.ui.depComboboxIon.setEnabled(False)

	def onDepNormRadioToggle(self):
		if self.ui.depRadioIon.isChecked():
			self.ui.depComboboxIon.setEnabled(True)
		else:
			self.ui.depComboboxIon.setEnabled(False)

	def onDeployAggCheck(self):
		if self.ui.deployAGGcheck.isChecked():
			State = True
		else:
			State = False
		self.ui.aggwLabel.setEnabled(State)
		self.ui.aggmodeLabel.setEnabled(State)
		self.ui.aggW.setEnabled(State)
		self.ui.aggMode_2.setEnabled(State)
		
	def onDepMaskcheck(self):
		if self.ui.depMaskcheck.isChecked():
			state = True
		else:
			state = False
		self.ui.depGoVisButton.setEnabled(state)
		self.ui.depVisSelLabel.setEnabled(state)
		self.ui.depVisListCombo.setEnabled(state)
		self.ui.depGoSegEdButton.setEnabled(state)
		self.ui.depSegListLabel.setEnabled(state)
		self.ui.depSegListCombo.setEnabled(state)

	def onDepGoVis(self):
		self.ui.tabWidget.setCurrentIndex(1)

	def onDepGoSeg(self):
		sourceVolumeNode = slicer.util.getNode( self.ui.depVisListCombo.currentText )
		slicer.util.selectModule("SegmentEditor")

		# set master volume and geometry
		segmentEditorNode = slicer.util.getNodesByClass('vtkMRMLSegmentEditorNode')[0]
		segmentationNode = slicer.util.getNodesByClass('vtkMRMLSegmentationNode')[0]
		segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(sourceVolumeNode)
		segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
		segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
		segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
		segmentEditorWidget.setSegmentationNode(segmentationNode)
		segmentEditorWidget.setSourceVolumeNode(sourceVolumeNode)

		# set the display view and link views
		RedCompNode = slicer.util.getNode("vtkMRMLSliceCompositeNodeRed")
		RedCompNode.SetBackgroundVolumeID(sourceVolumeNode.GetID())
		RedNode = slicer.util.getNode("vtkMRMLSliceNodeRed")
		RedNode.SetOrientation("Axial")

		slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)
		slicer.util.resetSliceViews()

	def onDepSegListUpdate(self):
		segmentationNode = slicer.util.getNodesByClass('vtkMRMLSegmentationNode')[0]
		segmentation = segmentationNode.GetSegmentation()
		segIDs = segmentation.GetSegmentIDs()
		segNames = [segmentation.GetSegment(segID).GetName() for segID in segIDs]

		self.ui.depSegListCombo.clear()
		self.ui.depSegListCombo.addItem('None')
		for segName in segNames:
			self.ui.segVollist1.addItem(segName)

	def onApplyDeployment(self):
		# spectrum normalization
		if self.ui.deployNormcheck.isChecked():
			if self.ui.depRadioTIC.isChecked():
				spec_normalization = 'tic'
			else:
				spec_normalization = self.ui.depComboboxIon.currentText
		else:
			spec_normalization = None
	
		# spectrum aggregation
		if self.ui.deployAGGcheck.isChecked():
			pixel_aggregation = ( int(self.ui.aggW.text), self.ui.aggMode_2.currentText )
		else:
			pixel_aggregation = None
		
		dep_mask = None
		if self.ui.depMaskcheck.isChecked():
			dep_mask = self.ui.depSegListCombo.currentText

		self.logic.model_deployment(spec_normalization, pixel_aggregation, dep_mask)

	### Boilerplate functions from template
			
	def cleanup(self):
		"""
		Called when the application closes and the module widget is destroyed.
		"""
		self.removeObservers()

	def enter(self):
		"""
		Called each time the user opens this module.
		"""
		# Make sure parameter node exists and observed
		self.initializeParameterNode()

	def exit(self):
		"""
		Called each time the user opens a different module.
		"""
		# Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
		self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

	def onSceneStartClose(self, caller, event):
		"""
		Called just before the scene is closed.
		"""
		# Parameter node will be reset, do not use it anymore
		self.setParameterNode(None)

	def onSceneEndClose(self, caller, event):
		"""
		Called just after the scene is closed.
		"""
		# If this module is shown while the scene is closed then recreate a new parameter node immediately
		if self.parent.isEntered:
			self.initializeParameterNode()

	def initializeParameterNode(self):
		"""
		Ensure parameter node exists and observed.
		"""
		# Parameter node stores all user choices in parameter values, node selections, etc.
		# so that when the scene is saved and reloaded, these settings are restored.

		self.setParameterNode(self.logic.getParameterNode())

		# Select default input nodes if nothing is selected yet to save a few clicks for the user
		if not self._parameterNode.GetNodeReference("InputVolume"):
			firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
			if firstVolumeNode:
				self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

	def setParameterNode(self, inputParameterNode):
		"""
		Set and observe parameter node.
		Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
		"""

		if inputParameterNode:
			self.logic.setDefaultParameters(inputParameterNode)

		# Unobserve previously selected parameter node and add an observer to the newly selected.
		# Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
		# those are reflected immediately in the GUI.
		if self._parameterNode:
			self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
		self._parameterNode = inputParameterNode
		if self._parameterNode:
			self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

		# Initial GUI update
		self.updateGUIFromParameterNode()

	def updateGUIFromParameterNode(self, caller=None, event=None):
		"""
		This method is called whenever parameter node is changed.
		The module GUI is updated to show the current state of the parameter node.
		"""

		if not self._parameterNode or self._updatingGUIFromParameterNode:
			return

		# Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
		self._updatingGUIFromParameterNode = True

		# All the GUI updates are done
		self._updatingGUIFromParameterNode = False

	def updateParameterNodeFromGUI(self, caller=None, event=None):
		"""
		This method is called when the user makes any change in the GUI.
		The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
		"""

		if not self._parameterNode or self._updatingGUIFromParameterNode:
			return

		wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

		self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
		self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)
		self._parameterNode.SetParameter("Threshold", str(self.ui.imageThresholdSliderWidget.value))
		self._parameterNode.SetParameter("Invert", "true" if self.ui.invertOutputCheckBox.checked else "false")
		self._parameterNode.SetNodeReferenceID("OutputVolumeInverse", self.ui.invertedOutputSelector.currentNodeID)

		self._parameterNode.EndModify(wasModified)

	def onApplyButton(self):
		"""
		Run processing when user clicks "Apply" button.
		"""
		try:
			# Compute output
			self.logic.process(self.ui.inputSelector.currentNode(), 
												 self.ui.outputSelector.currentNode(),
												 self.ui.imageThresholdSliderWidget.value, 
												 self.ui.invertOutputCheckBox.checked)
			# Compute inverted output (if needed)
			if self.ui.invertedOutputSelector.currentNode():
				# If additional output volume is selected then result with inverted threshold is written there
				self.logic.process(self.ui.inputSelector.currentNode(), 
													 self.ui.invertedOutputSelector.currentNode(),
													 self.ui.imageThresholdSliderWidget.value, 
													 not self.ui.invertOutputCheckBox.checked, 
													 showResult=False)
		except Exception as e:
			slicer.util.errorDisplay("Failed to compute results: "+str(e))
			import traceback
			traceback.print_exc()

	def onReload(self):
		"""
		Modified the classic reload button since splitting the module
		into files requires a custom function. Code is taken from:
		https://discourse.slicer.org/t/python-scripted-module-development-reload-feature-for-multiple-files/6363/4 
		"""
		logging.debug("Reloading MassVision")
		packageName='MassVisionLib'
		submoduleNames=['Logic', 'Utils']
		import imp
		f, filename, description = imp.find_module(packageName)
		package = imp.load_module(packageName, f, filename, description)
		for submoduleName in submoduleNames:
			f, filename, description = imp.find_module(submoduleName, package.__path__)
			try:
				imp.load_module(packageName+'.'+submoduleName, f, filename, description)
			finally:
				f.close()
		ScriptedLoadableModuleWidget.onReload(self)
