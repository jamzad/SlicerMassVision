from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, roc_auc_score, recall_score
import numpy as np
import matplotlib.pyplot as plt

'''
    Utils for model training.
    Author: Ayesha Syeda
	Updated by: Amoon Jamzad
'''

def saveTrainingPlot(y, x, title, dir_checkpoint):
	plt.figure()
	plt.plot(x, y)
	plt.title(title)
	plt.xlabel('Epoch')
	plt.ylabel(title)
	plt.xticks(np.arange(min(x), max(x)+1, 1.0))
	plt.savefig(f'{dir_checkpoint}/{title}.jpeg')

def get_fold_info(y_train, y_test, class_order, split_mode):
	fold_string = '#### DATA DISTRIBUTION ####################\n'
	fold_string += '#### train set: \n'
	fold_string += get_information_str(y_train, class_order)
	if split_mode!= 'all_train':
		fold_string += '\n#### test set: \n'
		fold_string += get_information_str(y_test, class_order)
	return fold_string

def get_performance_info(y_train, y_train_preds, y_train_prob, 
			 			y_test, y_test_preds, y_test_prob,
						class_order, split_mode):
	performance_string = '#### MODEL PERFORMANCE ####################\n'
	performance_string += '#### train set: \n'
	performance_string += get_performance_str(y_train, y_train_preds, y_train_prob, class_order)
	if split_mode!='all_train':
		performance_string += '\n#### test set: \n'
		performance_string += get_performance_str(y_test, y_test_preds, y_test_prob, class_order)
	return performance_string
	

def get_information_str(y, classnames):
	info_str = ''
	total_data = 0
	for name in classnames:
		count = np.count_nonzero(y == name)
		total_data += count
		info_str += f'{name}: {count}\n'
	info_str += f'total: {total_data}\n'
	return info_str
	
def get_performance_str(y_train, y_train_preds, y_train_prob, class_order):
	conf_matrix = confusion_matrix(y_train, y_train_preds, labels=class_order)
	results_str = f"confusion matrix: \n{conf_matrix}\n"
	
	cnf = ''
	for i in range(len(class_order)):
		for val in conf_matrix[i]:
			cnf += f'{val}\t'
		cnf += f'{class_order[i]}\n'
	print(cnf)

	acc = accuracy_score(y_train, y_train_preds)
	results_str += f"accuracy: {np.round(100*acc,2)}\n"

	bac = balanced_accuracy_score(y_train, y_train_preds)
	results_str += f"balanced accuracy: {np.round(100*bac,2)}\n"

	if len(set(y_train)) == len(set(class_order)):
		
		if len(class_order)==2: #binary
			recall_all = recall_score(y_train, y_train_preds, average=None)
			# results_str += f"specificity: {np.round(100*recall_all[0],2)}\n"
			# results_str += f"sensitivity: {np.round(100*recall_all[1],2)}\n"
			for i in range(len(class_order)):
				results_str += f"{class_order[i]} recall/sensitivity: {np.round(100*recall_all[i],2)}\n"
			auc = roc_auc_score(y_train, y_train_prob[:,-1], average='macro')
			results_str += f"AUC: {np.round(auc,2)}\n"
		else:
			recall_all = recall_score(y_train, y_train_preds, average=None)
			for i in range(len(class_order)):
				results_str += f"{class_order[i]} recall/sensitivity: {np.round(100*recall_all[i],2)}\n"
			auc = roc_auc_score(y_train, y_train_prob, average='macro', multi_class='ovr')
			results_str += f"AUC: {np.round(auc,2)}\n"

	else:

		for lab in np.sort(list(set(y_train))):
			class_recall = recall_score(y_train, y_train_preds, labels=[lab], average=None)
			results_str += f"{lab} recall/sensitivity: {np.round(100*class_recall[0],2)}\n"

	# if len(set(y_train)) == len(set(class_order)):


	# if len(classnames) < 2:
	# 	results_str += 'No confusion matrix available (< 2 labels in the dataset)\n'
	# if len(classnames) >= 2: # binary classification
	# 	conf_matrix = confusion_matrix(y_labels, y_pred, labels=classnames)
	# 	cnf = ''
	# 	for i in range(len(classnames)):
	# 		for val in conf_matrix[i]:
	# 			cnf += f'{val}\t'
	# 		cnf += f'{classnames[i]}\n'
	# 	print(cnf)
	# 	# results_str += f"Confusion matrix: \n{cnf}"
	# 	results_str += f"Confusion matrix: \n{conf_matrix}\n"
	# acc = accuracy_score(y_labels, y_pred)
	# bac = balanced_accuracy_score(y_labels, y_pred)
	# results_str += f"  Accuracy: {np.round(100*acc,2)}\n"
	# results_str += f"  Balanced accuracy: {np.round(100*bac,2)}\n"
	# results_str += f"\n"
	# # TN, FN, FP, TP = conf_matrix.ravel()
	# # sensitivity = TP/(TP+FN) if TP+FN > 0 else 'N/A'
	# # specificity = TN/(TN+FP) if TN+FP > 0 else 'N/A'
	# # bal_acc = (sensitivity + specificity)/2 if sensitivity != 'N/A' and specificity != "N/A" else "N/A"
	# # results_str += f"\u2022Balanced Accuracy: {bal_acc}\n"

	# # precision, recall, f1score =  precision_recall_fscore_support(y_labels, y_pred, labels=[cancerClass])[:3]
	# # results_str += f"\u2022Precision: {precision.item()}\n"
	# # results_str += f"\u2022Recall: {recall.item()}\n"
	# # results_str += f"\u2022F1-score: {f1score.item()}\n\n"
	
	return results_str
