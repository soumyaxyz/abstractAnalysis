def split_data(threshold, data_file):
	output_file 	= '/users/debarshi/soumya/arxiv_final/arxiv_selected.txt'

	
	infile 				= open(data_file,'r')
	no_of_entry_copied 	= 0
	lines_in_abstract	= -5 # meta dtata and black lines
	buffered_abstract	= ''
	outfile 			=  open(output_file,'w+') 
	for line in infile:	
		if line.startswith('### '):
			if lines_in_abstract > threshold:
				# print(buffered_abstract)	
				outfile.write(buffered_abstract)
				no_of_entry_copied += 1
			buffered_abstract	= ''
			lines_in_abstract	=  -5 # meta dtata and black lines

			buffered_abstract += "### "+str(no_of_entry_copied+1)+'\n'
		else:
			buffered_abstract += line
		lines_in_abstract += 1 
		
		
	print ('Wrote '+ str(no_of_entry_copied) + ' abstracts to '+ output_file)
	outfile.close()
		


import sys
args = sys.argv

try:
	threshold =  float(args[1])
except Exception as e:
	threshold 	= 9
try:
	data_file =  int(args[2])
except Exception as e:
	data_file 	= "/users/debarshi/soumya/abstractAnalysis/cs.NI.txt"


     
split_data(threshold, data_file)