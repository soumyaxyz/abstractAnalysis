import numpy as np, seaborn as sns, pandas as pd
from collections import Counter 
import matplotlib.pyplot as plt
import termplot, pprint, pdb, traceback
from  sklearn.metrics import cohen_kappa_score as kappa



def plot_distribution(line_label, line_pred, avg_num_of_lines):
	labels= ['BACK_GOLD','BACK_PRED','DESC_GOLD','DESC_PRED', 'OBSV_GOLD', 'OBSV_PRED']
	pp = pprint.PrettyPrinter(indent=4)

	normalized_line_label = np.array([line / line.sum() for line in line_label])
	normalized_line_label =  np.transpose(normalized_line_label)
	pp.pprint(line_label)

	normalized_line_pred = np.array([line / line.sum() for line in line_pred])
	normalized_line_pred =  np.transpose(normalized_line_pred)
	pp.pprint(line_pred)

	colors=['#DAF7A6', '#FFC300', '#FF5733', '#C70039', '#900C3F', '#581845']
	# pdb.set_trace()
		

	ind = [*range(avg_num_of_lines)]
	ind1 = [i+.4 for i in range(9)]
	barWidth = .4
	for i, line in enumerate(normalized_line_label):
		# print(i,' ',normalized_line_label[i])
		if i==0:
			plt.barh(ind, normalized_line_label[i], color=colors[i], edgecolor='white', height=barWidth, label=labels[i] )
			plt.barh(ind1, normalized_line_pred[i], color=colors[i+3], edgecolor='white', height=barWidth, label=labels[i] )
		else:
			plt.barh(ind, normalized_line_label[i] ,left=np.sum(normalized_line_label[:i], axis=0), color=colors[i], edgecolor='white', height=barWidth, label=labels[i])
			plt.barh(ind1, normalized_line_pred[i] ,left=np.sum(normalized_line_pred[:i], axis=0), color=colors[i+3], edgecolor='white', height=barWidth, label=labels[i])
	
	# plt.xticks()
	plt.xlim(0,1.04)
	plt.ylim(-.6,9)
	plt.yticks([i+.2 for i in range(0, 9)],range(1, 10))
	plt.legend(labels, bbox_to_anchor=(1.1, 1.16), ncol=3)
	plt.gca().invert_yaxis()	
	plt.savefig('result_distribution.png')
	plt.show()

	### rewriten with pandas
	# try:			
		# normalized_line_label = np.array([line / line.sum() for line in line_label])
		# df = pd.DataFrame(normalized_line_label, columns=['BACKGROUND','DESCRIPTION','OBSERVATION'] )
		# df.plot(kind='bar', stacked=True)
		# plt.show()
	# except Exception as e:
	# 	traceback.print_exc()
	# 	pdb.set_trace()	



def count(test_file, pred_file):
	pp = pprint.PrettyPrinter(indent=4) 

	remap = [-1,0,1,2]
	

	labels = ['BACKGROUND','DESCRIPTION','OBSERVATION'] 
	lines = open(test_file, 'r', encoding="utf8").readlines()
	preds = open(pred_file, 'r', encoding="utf8").readlines()
	lines.append('###')

	
	
	count 			= Counter()
	abstract_size	= Counter()
	avg_num_of_lines = 9
	line_label 		= np.zeros((avg_num_of_lines,3))
	line_pred 		= np.zeros((avg_num_of_lines,3))


	

	label_list = []
	abstracts = []

	abs_count = 0
	i = 0
	new_abstract = True
	for l, line in enumerate(lines):
		line = line.strip()

		if not line or line.startswith('#'):
			if new_abstract:
				abs_count   += 1
				abstract 	= []
				prev 		= -1
				curr 		= -1
				if label_list:
					# print('_____\n',label_list,'_____')
					scale_factor = avg_num_of_lines/abstract_len
					# for j in range(abstract_len):
					# 	# print(j,' > ',int(j*scale_factor) ," : ", label_list[j] )
					# 	
					score = 0
					for l1, label in enumerate(label_list):
						try:
							pred = [remap[int(i)] for i in preds[i].split(",")]
						except IndexError:
							break
						except Exception:
							traceback.print_exc()
							pdb.set_trace()	
						
						i 	+=1
						if pred[1] != int(label):
							pdb.set_trace()	
						else:
							# print(labels[label], labels[pred[0]], line_list[l1] )
							abstract.append([labels[pred[0]], labels[pred[1]], line_list[l1]])
							line_label[int(l1*scale_factor)][label_list[l1]] += 1
							line_pred[int(l1*scale_factor)][pred[0]] += 1
							if pred[0] == pred[1]:
								score += 1


					score = round(score*100/abstract_len,2)
					print(abs_count-2, score)
					abstracts.append([abstract, score])


					# print('_____\n',label_list,'_____')
				label_list 	= []
				line_list 	= []

				try:					
					# if abstract_size[abstract_len] == 0: 
					# 	print(abstract_len, abstract_size)
					abstract_size[abstract_len] += 1
				except Exception as e:
					pass
				abstract_len = 0
			new_abstract = False
		else: 
			new_abstract = True
			prev 	= curr
			try:
				arr = line.split(" ", 1)
			except Exception as e:
				arr	= line.split("\t", 1)

			curr 	= int(arr[0])
			text	= arr[1]
			

			abstract_len			+=1
			count[labels[curr]] 	+=1

			# print(abstract_len,' > ', curr)
			label_list.append(curr)
			line_list.append(text)
			# line_label[]

			
					
	# abstract_size[abstract_len] += 1

	plot_distribution(line_label, line_pred, avg_num_of_lines)

	print('number of abstracts :', abs_count-1)

	# print('abstract_len :',abstract_size)

	
	# plt.bar(abstract_size.keys(), abstract_size.values())
	# plt.plot([*abstract_size.keys()], [*abstract_size.values()])

	idx = sorted (abstract_size)
	nun_lines = [ abstract_size[i] for i in idx]
	# pdb.set_trace()	

	# plt.plot(idx, nun_lines  )
	# plt.show()
	wt_avg = sum(idx[i] * nun_lines[i] / sum(nun_lines) for i in range(len(idx)))

	print('average number of lines per abstracts : ', wt_avg)

	print('total number of lines :', sum(count.values()))
	# pdb.set_trace()

	
def kappa_calc(file1, file2):

	lines 	= open(file1, 'r', encoding="utf8").readlines()
	lines2 	= open(file2, 'r', encoding="utf8").readlines()
	lines.append('###')

	
	labels1 = []
	labels2 = []

	

	abs_count = 0
	i = 0
	new_abstract = True
	for l, line in enumerate(lines):
		line = line.strip()

		if not line or '#' in line:
			pass
		else: 
			try:
				
			
				try:
					l1 	= int(line.split(" ", 1)[0])
				except Exception as e:
					l1	= int(line.split("\t", 1)[0])
				try:
					l2 	= int(lines2[l].split(" ", 1)[0])
				except Exception as e:				
					l2 	= int(lines2[l].split("\t", 1)[0])

			except Exception as e:
				traceback.print_exc()
				pdb.set_trace()

			
			labels1.append(l1)
			labels2.append(l2)

	kappa(labels1,labels2)		
					
	# abstract_size[abstract_len] += 1

	

	
	print('The cohen kappa score is : ', kappa(labels1,labels2))



	

	
#main function
if __name__ == "__main__":

	# test_file = '../IEEE_final/TLT/test_clean.txt'
	# pred_file = '20k_TLT/model_Jin_2k_IEEE_TLT_110.csv'
	# count(test_file, pred_file)

	file_1 = 'arxiv_rest.txt'
	file_2 = 'pred_arxiv_rest.txt'
	kappa_calc(file_1, file_2)