# Basic CSV IO
import os

def read_data(file_name, skipFirstLine = True, split = ","):
	f = open(file_name)

		
	samples = []
	target = []

	for line in f.readlines():
		if ( skipFirstLine ): 
			skipFirstLine = False
			continue
		line = line.strip().replace("|", ",").split(split)
		sample = [x.replace("\"", "") for x in line]
		for item in sample:

			if ( item.replace('.','',1).isdigit() ) :
				if ( item == "1" or item == "0"):
					item = float(item);
				else:
					item = float(item)

		samples.append(sample)
	return samples

def delete_file(file_path):
	if ( file_path.startswith( "c:" ) or file_path.startswith( "\\" ) or file_path.startswith( "*" )):
		return
	os.unlink(filename)
	
	
def write_delimited_file(file_path, data,header=None, delimiter=",", filemode="w"): # filemode can be "a" for append
	f_out = open(file_path,filemode)
	if header is not None:
		f_out.write(delimiter.join(header) + "\n")
	for line in data:
		
		if isinstance(line, str):
			f_out.write(line + "\n")
		else:
			output = ""
			delim = ""
			for item in line:
				if isinstance(item, dict):
					keys = sorted(list(item.keys()))
					for key in keys:
						output += delim + str(item[key])
						
					delim = delimiter	
				else:
					output += delim + str(item)
				
				delim = delimiter

			f_out.write(output+ "\n")
	f_out.close()
	
def write_delimited_file_GUID(file_path, Guid_path, data,header=None, delimiter=","):
	f_out = open(file_path,"w")
	
	GuidArray = read_data(Guid_path, False)
	
	if header is not None:
		f_out.write(delimiter.join(header) + "\n")
		
		
	GuidIndex = 0	
	for line in data:
		
		if isinstance(line, str):
			f_out.write(str(GuidArray[GuidIndex][0]) + "," + line + "\n")
		else:
			line = [str(x) for x in line]
			line.insert(GuidArray[GuidIndex][0], 0)
			f_out.write(delimiter.join(line) + "\n")
		
		GuidIndex = GuidIndex + 1
	f_out.close()
	
	
def write_delimited_file_GUID_numpy(file_path, Guid_path, data,header=None, delimiter=","):
	f_out = open(file_path,"w")
	
	GuidArray = read_data(Guid_path, False)
	
	if header is not None:
		f_out.write(delimiter.join(header) + "\n")
		
		
	GuidIndex = 0	
	for line in data:
		
		if isinstance(line, str):
			f_out.write(str(GuidArray[GuidIndex][0]) + "," + line + "\n")
		else:
			lineOut = [];
			lineOut.append(str(GuidArray[GuidIndex][0]))
			lineOut.append(str(line))

			f_out.write(delimiter.join(lineOut) + "\n")
		
		GuidIndex = GuidIndex + 1
	f_out.close()