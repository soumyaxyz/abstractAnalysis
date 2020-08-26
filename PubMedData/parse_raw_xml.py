import os 
import xml.etree.ElementTree as ET
from spacy.lang.en import English

# python 3

def write_to(outfile, string, as_unicode = True):
	if as_unicode:
		# uniString = unicode(string, "utf-8" ,errors='ignore' ).encode("utf-8")
		outfile.write(string) #.encode('utf-8',errors='ignore') )
	else:
		outfile.write(string ) #.encode('ascii',errors='ignore') )



def sanitize_label(label, classFlags):
	LABEL = label.upper()
	classFlags
	if 'OBJECTIVE' in LABEL:
		classFlags[1] = 1
		LABEL = 'OBJECTIVE  '
	elif 'METHODS' in LABEL:
		classFlags[2] = 1
		LABEL = 'METHODS    '
	elif 'RESULTS' in LABEL:
		classFlags[3] = 1
		LABEL = 'RESULTS    '
	elif 'BACKGROUND' in LABEL:
		classFlags[0] = 1
		LABEL = 'BACKGROUND '
	elif 'CONCLUSIONS' in LABEL:
		classFlags[4] = 1
		LABEL = 'CONCLUSIONS'
	else:
		classFlags[5] = 1
	return (LABEL, classFlags)


nlp         	= English()
sentencizer 	= nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

path = 'raw_xml'
# outfile_name = os.path.join('data','pubmed_structured_data.txt')
# three_or_more_class_outfile_name = os.path.join('data','pubmed_structured_5_class_data.txt')


save_as_unicode = True
f = []
for (_, _, filenames) in os.walk(path):
	f.extend(filenames)
	break
outfile_name 						= os.path.join('data','all','pubmed_structured_data.txt')
three_or_more_class_outfile_name 	= os.path.join('data','three_or_more','pubmed_structured_3_or_more_class_data.txt')
count 		= 0
count_3 	= 0


outfile    				= open(outfile_name,'w+')
three_or_more_class_outfile 		= open(three_or_more_class_outfile_name,'w+')

for file in filenames:
	# fileid = int(file[9:13])
	# print(fileid)
	# if fileid < 41:
	# 	continue

	try:		
		xmlFile 				= os.path.join(path,file)
		print('\tInitializing '+xmlFile)
		# import pdb; pdb.set_trace()
		tree = ET.parse(xmlFile)
		root = tree.getroot()
		print('\tParsing '+xmlFile)

		# print(root.tag)
		for child in root:
			for gchild in child:
				# print(gchild.tag+'\t'+str(gchild.attrib))
				if 'MedlineCitation' in gchild.tag:
					flag 	= False
					flag_3 	= False
					title  	= ''
					for ggchild in gchild:
						# print('\t'+ggchild.tag+'\t'+str(ggchild.attrib))	
						if 'Article' in ggchild.tag:
							title = ggchild[1].text					
							for gggchild in ggchild:
								# print('\t\t\t'+gggchild.tag+'\t'+str( gggchild.attrib)
								# flag = False
								# flag_3 = False
								if 'Abstract' in gggchild.tag:
									abstract_text_lines = []									
									classFlags = [0,0,0,0,0,0]
									for abstract_text in gggchild:
										if abstract_text.attrib != {}:
											label = ''
											try:
												label = str( abstract_text.attrib['Label'])
											except :
												pass#print (abstract_text.attrib)
											# import pdb; pdb.set_trace()
											try:
												label += str( abstract_text.attrib['NlmCategory'])
											except :
												pass#print (abstract_text.attrib)
											if 	label == '':
												label = str(abstract_text.attrib)
											# import pdb; pdb.set_trace()
											(label, classFlags) 	= sanitize_label(label, classFlags)
											abstract_text_content 	= abstract_text.text
											if abstract_text_content is not None:
												abstract_part 			= nlp(abstract_text_content)
												for line in abstract_part.sents:
													line = str(line).strip()
													abstract_text_lines.append(label+'\t\t\t\t'+ str(line))
												flag = True
												if sum(classFlags[:4]) >= 3 and classFlags[5] == 0:
													flag_3 = True	
								if flag and 'PublicationTypeList' in gggchild.tag:
									for PublicationType in gggchild:
										# print(gggchild.tag, gggchild.attrib)
										# print('\t'+ggchild.tag, ggchild.attrib)
										# print('\t\t',gggchild.tag, gggchild.attrib)
										# print('\t\t\t',PublicationType.tag, PublicationType.attrib, PublicationType.text)
										# import pdb; pdb.set_trace()
										if PublicationType.attrib['UI'] == 'D016449':											
											# print('\t\t\t',PublicationType.tag, PublicationType.attrib, PublicationType.text,'X')
											flag = False
											flag_3 = False
											break										

					if flag:
						count +=1
						write_to(outfile, '### '+gchild[0].text+'\n', save_as_unicode)
						write_to(outfile, '#### '+title+'\n\n', save_as_unicode)
						# print len (abstract_text_lines)
						for line in abstract_text_lines:
							# line = line.encode('ascii',errors='ignore')
							write_to(outfile, line+'\n', save_as_unicode)
						write_to(outfile, '\n\n\n', save_as_unicode)
						# print(gggchild.tag, gggchild.attrib)
						# print('\t'+ggchild.tag, ggchild.attrib)
						# print('\t\t',gggchild.tag, gggchild.attrib)
							# import pdb; pdb.set_trace()
					if flag_3:
						count_3 +=1
						write_to(three_or_more_class_outfile, '### '+gchild[0].text+'\n', save_as_unicode)
						write_to(three_or_more_class_outfile, '#### '+title+'\n\n', save_as_unicode)
						# print len (abstract_text_lines)
						for line in abstract_text_lines:
							# line = line.encode('ascii',errors='ignore')
							write_to(three_or_more_class_outfile, line+'\n', save_as_unicode)
						write_to(three_or_more_class_outfile, '\n\n\n', save_as_unicode)
	except Exception as e:
			import traceback
			traceback.print_exc()
			import pdb; pdb.set_trace()
	print('\tSaved data from '+xmlFile)
	print('\tFound '+str(count)+' structured non RCT abstract')
	print('\tFound '+str(count_3)+' structured non RCT abstract with three_or_more_class')

write_to(outfile, '### count = '+str(count) , save_as_unicode)
outfile.close()
write_to(three_or_more_class_outfile, '### count = '+str(count_3) , save_as_unicode)
three_or_more_class_outfile.close()
