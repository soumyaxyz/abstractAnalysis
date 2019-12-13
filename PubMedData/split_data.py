def split_data(total_data, train, dev, test):
	data_file 		= '/users/soumya/PubMedData/data/three_or_more/pubmed_structured_3_or_more_class_data.txt'
	output_folder 	= '/users/soumya/PubMedData/output/'

	frac = total_data/ (train +  dev + test)

	dev_size 	= int(round(frac * dev))
	test_size 	= int(round(frac * test))
	train_size 	= total_data - (dev_size + test_size )

	# print(train_size , dev_size , test_size , train_size + dev_size + test_size, total_data )

	infile 				= open(data_file,'r')
	no_of_entry_copied 	= 0
	i 					= 0
	size 				= [dev_size, test_size, train_size]
	outfile_names 		= ['dev_clean', 'test_clean', 'train_clean']
	buffered_abstract	= ''
	outfile 			=  open(output_folder+outfile_names[i]+'.txt','w+') 
	for line in infile:	
		if line.startswith('### '):						
			outfile.write(buffered_abstract)
			buffered_abstract	= ''
			no_of_entry_copied += 1
		if no_of_entry_copied == size[i]:
			print ('Wrote '+ str(no_of_entry_copied) + ' abstracts to '+ output_folder+outfile_names[i]+'.txt')
			no_of_entry_copied 	= 0
			i 					+= 1
			if i == 3:
				break
			outfile.close()
			outfile 			=  open(output_folder+outfile_names[i]+'.txt','w+') 

		buffered_abstract += line


import sys
args = sys.argv

try:
	total_data =  int(args[1])
except Exception as e:
	total_data 	= 20000
try:
	train =  float(args[2])
except Exception as e:
	train 	= 75
try:
	dev =  float(args[3])
except Exception as e:
	dev 	= 12.5
try:
	test =  float(args[4])
except Exception as e:
	test 	= 12.5

# print (total_data, train, dev, test) 

     
split_data(total_data, train, dev, test)