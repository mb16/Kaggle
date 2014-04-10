#!/opt/local/bin python2.7

import csv_io
import math
from math import log
import string
import copy
import operator

def toFloat(str):
	return float(str)

def PreProcess():
	PreProcessRun("training")
	PreProcessRun("test")	
	
def PreProcessRun(dataSet):
	print
	print "DataSet: ", dataSet
	
	DataClassList = ["Gender", "Age", "height", "weight", "bmi", "systolicBP", "diastolicBP"]

	
	
	# since using file append, must delete first.
	print "Hey, need to delete the _PreProcess.csv files first."
	#csv_io.delete_file("PreProcessData/" + dataSet + "_PreProcess.csv")
	#return
	
	SyncPatient = csv_io.read_data("../" + dataSet + "Set/" + dataSet + "Set/" + dataSet + "_SyncPatient.csv")
	SyncPatientSorted = sorted(SyncPatient, key=lambda patient: patient[0])
	#print SyncPatientSorted

	
	SyncTranscript = csv_io.read_data("../" + dataSet + "Set/" + dataSet + "Set/" + dataSet + "_SyncTranscript.csv")
	SyncTranscriptSorted = sorted(SyncTranscript, key=lambda patient: patient[1])
	#print SyncTranscriptSorted

	
	SyncPatientSmokingStatus = csv_io.read_data("../" + dataSet + "Set/" + dataSet + "Set/" + dataSet + "_SyncPatientSmokingStatus.csv")
	SyncPatientSmokingStatusSorted = sorted(SyncPatientSmokingStatus, key=lambda patient: patient[1])
	#print SyncPatientSmokingStatusSorted
	SmokingStatusList = {}
	for Record in SyncPatientSmokingStatusSorted:
		if Record[2] not in SmokingStatusList:
			SmokingStatusList[Record[2]] = 0;
			
	
	DiagnosisList = {}
	SyncDiagnosis = csv_io.read_data("../testSet/testSet/test_SyncDiagnosis.csv")
	for Record in SyncDiagnosis:
		if Record[2] not in DiagnosisList:
			DiagnosisList[Record[2]] = 0;
		else:
			DiagnosisList[Record[2]] += 1
	SyncDiagnosis = csv_io.read_data("../trainingSet/trainingSet/training_SyncDiagnosis.csv")
	for Record in SyncDiagnosis:
		if Record[2] not in DiagnosisList:
			DiagnosisList[Record[2]] = 0;
		else:
			DiagnosisList[Record[2]] += 1
	

	# remove from diagnosis list items occuring less than 50 times...
	for key in DiagnosisList.keys():
		if DiagnosisList[key] < 30:
			del DiagnosisList[key]
			
	#print sorted(DiagnosisList.iteritems(), key=operator.itemgetter(1))
	#print len(DiagnosisList)
	
	SyncDiagnosis = csv_io.read_data("../" + dataSet + "Set/" + dataSet + "Set/" + dataSet + "_SyncDiagnosis.csv")
	SyncDiagnosisSorted = sorted(SyncDiagnosis, key=lambda patient: patient[1])		


	# diagnosis with decimal digits removed.
	DiagnosisListX = {}
	SyncDiagnosis = csv_io.read_data("../testSet/testSet/test_SyncDiagnosis.csv")
	for Record in SyncDiagnosis:
		Record2 = Record[2].split(".")[0]
		if Record2 not in DiagnosisListX:
			DiagnosisListX[Record2] = 0;
		else:
			DiagnosisListX[Record2] += 1
	SyncDiagnosis = csv_io.read_data("../trainingSet/trainingSet/training_SyncDiagnosis.csv")
	for Record in SyncDiagnosis:
		Record2 = Record[2].split(".")[0]
		if Record2 not in DiagnosisListX:
			DiagnosisListX[Record2] = 0;
		else:
			DiagnosisListX[Record2] += 1
	

	# remove from diagnosis list items occuring less than 50 times...
	for key in DiagnosisListX.keys():
		if DiagnosisListX[key] < 30:
			del DiagnosisListX[key]
			
	#print sorted(DiagnosisListX.iteritems(), key=operator.itemgetter(1))
	#print len(DiagnosisListX)

	
		
	MedicationList = {}
	SyncMedication = csv_io.read_data("../testSet/testSet/test_SyncMedication.csv")
	for Record in SyncMedication:
		first = Record[3].find("(")
		second = Record[3].find(")")
		medTuple = Record[3].partition(" ")
		med = medTuple[0].lower()
		if ( first != -1 and second != -1 ):
			med = Record[3][first+1:second].lower()
		if med not in MedicationList and med != "null":
			MedicationList[med] = 0;
		else:
			MedicationList[med] = 0;
		#print med
		#if Record[2] not in MedicationList:
			#MedicationList[Record[2]] = 0;
		#else:
			#MedicationList[Record[2]] += 1
	SyncMedication = csv_io.read_data("../trainingSet/trainingSet/training_SyncMedication.csv")
	for Record in SyncMedication:
		first = Record[3].find("(")
		second = Record[3].find(")")
		medTuple = Record[3].partition(" ")
		med = medTuple[0].lower()
		if ( first != -1 and second != -1 ):
			med = Record[3][first+1:second].lower()
		if med not in MedicationList and med != "null":
			MedicationList[med] = 0;
		else:
			MedicationList[med] = 0;
		#print med
		#if Record[2] not in MedicationList:
			#MedicationList[Record[2]] = 0;
		#else:
			#MedicationList[Record[2]] += 1
	

	# remove from Medication list items occuring less than 50 times...
	for key in MedicationList.keys():
		if MedicationList[key] < 30:
			del MedicationList[key]
			
	#print sorted(MedicationList.iteritems(), key=operator.itemgetter(1))
	#print len(MedicationList)
	#return

	SyncMedication = csv_io.read_data("../" + dataSet + "Set/" + dataSet + "Set/" + dataSet + "_SyncMedication.csv")
	SyncMedicationSorted = sorted(SyncMedication, key=lambda patient: patient[1])		
	
	
	LabObservationList = {}
	LabObservationDefaultValueList = {}
	SyncLabObservation = csv_io.read_data("../trainingSet/trainingSet/training_SyncLabObservation.csv", True, '","')
	#NOTE, if Record[0] is switched to "1", then a change needs to also occur in the lower section where these are processed.
	for Record in SyncLabObservation:
		if Record[0] != "NULL" and Record[5] != "NULL" and Record[5] != "":
			if Record[0] not in LabObservationList:
				#print "add", Record[0], Record
				LabObservationList[Record[0]] = 1
				LabObservationDefaultValueList[Record[0]] = float(Record[5])
			else:
				LabObservationList[Record[0]] += 1
				LabObservationDefaultValueList[Record[0]] += float(Record[5])
				
	# generate average values to fill in nulls....
	for LO in LabObservationList:
		#print LO, LabObservationDefaultValueList[LO], LabObservationList[LO]
		LabObservationDefaultValueList[LO] = float(LabObservationDefaultValueList[LO])/float(LabObservationList[LO]);
	
	#print LabObservationDefaultValueList
	#return
	#SyncLabObservation = csv_io.read_data("../testSet/testSet/test_SyncLabObservation.csv", True, '","' )
	#for Record in SyncLabObservation:
	#	if Record[0] != "NULL" and Record[5] != "NULL" and Record[5] != "":
	#		if Record[0] not in LabObservationList:
	#			print "found", Record[0]
	#			LabObservationList[Record[0]] = 1				
	#		else:
	#			LabObservationList[Record[0]] += 1
					
			

	# remove from LabObservation list items occuring less than 5 times...
	#for key in LabObservationList.keys():
	#	if LabObservationList[key] < 5:
	#		del LabObservationList[key]
			
	#print sorted(LabObservationTotalList.iteritems(), key=operator.itemgetter(1))
	#print len(LabObservationList)
	#return
	SyncLabObservation = csv_io.read_data("../" + dataSet + "Set/" + dataSet + "Set/" + dataSet + "_SyncLabObservation.csv", True, '","')
	

	SyncLabPanel = csv_io.read_data("../" + dataSet + "Set/" + dataSet + "Set/" + dataSet + "_SyncLabPanel.csv", True, '","')	
	#print SyncLabPanel
	#return
	
	LabPanelList = {}
	for Record in SyncLabPanel:
		#ii += 1 #60C9A966-AB92-441E-B6E7-3B0ABB7154A7
		#if Record[1].startswith("60C9A966"): #"2A39BFC5-2EBF-4A5D-8BA1-ED75196BAF4B" Record[1].startswith("2A39BFC5")
		#	print "FOUND", Record[1], Record[2]
		if Record[1] not in LabPanelList: # need to split because file format has strings and ints
			LabPanelList[Record[1]] = Record[2].split(",")[0]; # map from Lab Panel to LabResult 

	
	for Record in SyncLabObservation:
		#print Record[3], LabPanelList[Record[3]]
		Record[3] = LabPanelList[Record[3]] # substitute LabResultGuid for LabPanelGuid
		
	SyncLabResult = csv_io.read_data("../" + dataSet + "Set/" + dataSet + "Set/" + dataSet + "_SyncLabResult.csv", '","')	
	LabResultList = {}
	for Record in SyncLabResult:
		#print Record[0], Record[2]
		if Record[0] not in LabResultList: # need to split because file format has strings and ints
			#print Record[2]
			LabResultList[Record[0]] = Record[2]; # map from Lab Result to Patient GUID	

	missing = 0
	found = 0
	SyncLabObservationNew = [] # must skip the unlinked records.
	for Record in SyncLabObservation:
		if Record[3] not in LabResultList:
			missing += 1
			#print "Missing in SyncLabResult", Record[3]
		else:
			found += 1
			#print "Found"
			Record[3] = LabResultList[Record[3]] # substitute Patient for LabResultGuid 
			SyncLabObservationNew.append(Record)
	print "Found", found, "Missing" ,missing
	
	
	SyncLabObservationSorted = sorted(SyncLabObservationNew, key=lambda patient: patient[3])	# sort by column 3 which is not Patient GUID.	
	#print SyncLabObservationSorted
	#return
		
		
		
	LabPanelList = {}
	SyncLabPanel = csv_io.read_data("../testSet/testSet/test_SyncLabPanel.csv")
	for Record in SyncLabPanel:
		if Record[0] not in LabPanelList:
			LabPanelList[Record[0]] = 0;
		else:
			LabPanelList[Record[0]] += 1
	SyncLabPanel = csv_io.read_data("../trainingSet/trainingSet/training_SyncLabPanel.csv")
	for Record in SyncLabPanel:
		if Record[0] not in LabPanelList:
			LabPanelList[Record[0]] = 0;
		else:
			LabPanelList[Record[0]] += 1	
		
	#print LabPanelList
	#return
	SyncLabPanel = csv_io.read_data("../" + dataSet + "Set/" + dataSet + "Set/" + dataSet + "_SyncLabPanel.csv", True, '","')	
		
	missing = 0
	found = 0
	SyncLabPanelNew = [] # must skip the unlinked records.
	for Record in SyncLabPanel:
		#print Record[2].split(",")[0], Record
		if Record[2].split(",")[0] not in LabResultList:
			missing += 1
			#print "Missing in SyncLabResult", Record[3]
		else:
			found += 1
			#print "Found"
			Record[2] = LabResultList[Record[2].split(",")[0]] # substitute Patient for LabResultGuid 
			SyncLabPanelNew.append(Record)
	print "Found", found, "Missing" ,missing	
		
	SyncLabPanelSorted = sorted(SyncLabPanelNew, key=lambda patient: patient[2])	# sort by column 2 which is not Patient GUID.
		
		
		
		
	Output = []
	
	TranscriptIndex = 0
	SmokingStatusIndex = 0
	DiagnosisIndex = 0
	DiagnosisIndexX = 0
	MedicationIndex = 0
	LabObservationIndex = 0
	LabPanelIndex = 0
	
	HeightIsNone = 0
	WeightIsNone = 0
	BMIIsNone = 0
	SystolicBPIsNone = 0
	DiastolicBPIsNone = 0

	PatientGuidList = []
	

	
	# write list of data classes.
	for key in SmokingStatusList.keys():		
		DataClassList.append(key + "~Smoking")
	for key in DiagnosisList.keys():		
		DataClassList.append(key + "~Diagnosis")
	for key in DiagnosisListX.keys():		
		DataClassList.append(key + "~DiagnosisX")
	for key in MedicationList.keys():		
		DataClassList.append(key + "~Medication")
	for key in LabObservationList.keys():		
		DataClassList.append(key + "~LabObservation")
	for key in LabPanelList.keys():		
		DataClassList.append(key + "~LabPanel")
	DataClassList.append("Smoking1")
	DataClassList.append("Smoking2")
	DataClassList.append("Smoking3")
	DataClassList.append("Smoking4")
	DataClassList.append("Smoking5")
	DataClassList.append("Smoking6")
	DataClassList.append("SleepProblem")
	DataClassList.append("Candidiasis")
	DataClassList.append("MandOver20")
	DataClassList.append("MandOver25")
	DataClassList.append("MandOver30")
	DataClassList.append("MandOver35")
	DataClassList.append("MandOver40")
	DataClassList.append("MandOver45")
	DataClassList.append("MandOver50")
	DataClassList.append("MandOver55")
	DataClassList.append("MandOver60")
	DataClassList.append("MandOver65")
	DataClassList.append("MandOver70")
	DataClassList.append("MandOver75")
	DataClassList.append("FandOver20")
	DataClassList.append("FandOver25")
	DataClassList.append("FandOver30")
	DataClassList.append("FandOver35")
	DataClassList.append("FandOver40")
	DataClassList.append("FandOver45")
	DataClassList.append("FandOver50")
	DataClassList.append("FandOver55")
	DataClassList.append("FandOver60")
	DataClassList.append("FandOver65")
	DataClassList.append("FandOver70")
	DataClassList.append("FandOver75")
	csv_io.write_delimited_file("PreProcessData/DataClassList.csv", DataClassList)

	
	
	
	for Patient in SyncPatientSorted:
			
		PatientGuidList.append(Patient[0]) # used only for test data.	
			
		#print Patient[0], TranscriptIndex, len(SyncTranscriptSorted)
		GenderIndex = 2
		AgeIndex = 3
		if ( dataSet == "test" ):
			GenderIndex = 1
			AgeIndex = 2
			
		#print 	Patient[GenderIndex], float(Patient[AgeIndex])
			
		MandOver20 = 0
		MandOver25 = 0
		MandOver30 = 0
		MandOver35 = 0
		MandOver40 = 0
		MandOver45 = 0
		MandOver50 = 0
		MandOver55 = 0
		MandOver60 = 0
		MandOver65 = 0		
		MandOver70 = 0	
		MandOver75 = 0			
		FandOver20 = 0
		FandOver25 = 0
		FandOver30 = 0
		FandOver35 = 0
		FandOver40 = 0
		FandOver45 = 0
		FandOver50 = 0
		FandOver55 = 0
		FandOver60 = 0
		FandOver65 = 0		
		FandOver70 = 0	
		FandOver75 = 0	

	
		if  Patient[GenderIndex] == "M" and 2012.0 - float(Patient[AgeIndex]) > 20.0:
			MandOver20 = 1
		if  Patient[GenderIndex] == "M" and 2012.0 - float(Patient[AgeIndex]) > 25.0:			
			MandOver25 = 1
		if  Patient[GenderIndex] == "M" and 2012.0 - float(Patient[AgeIndex]) > 30.0:
			MandOver30 = 1
		if  Patient[GenderIndex] == "M" and 2012.0 - float(Patient[AgeIndex]) > 35.0:
			MandOver35 = 1
		if  Patient[GenderIndex] == "M" and 2012.0 - float(Patient[AgeIndex]) > 40.0:
			MandOver40 = 1
		if  Patient[GenderIndex] == "M" and 2012.0 - float(Patient[AgeIndex]) > 45.0:			
			MandOver45 = 1
		if  Patient[GenderIndex] == "M" and 2012.0 - float(Patient[AgeIndex]) > 50.0:	
			MandOver50 = 1
		if  Patient[GenderIndex] == "M" and 2012.0 - float(Patient[AgeIndex]) > 55.0:		
			MandOver55 = 1
		if  Patient[GenderIndex] == "M" and 2012.0 - float(Patient[AgeIndex]) > 60.0:		
			MandOver60 = 1
		if  Patient[GenderIndex] == "M" and 2012.0 - float(Patient[AgeIndex]) > 65.0:		
			MandOver65 = 1		
		if  Patient[GenderIndex] == "M" and 2012.0 - float(Patient[AgeIndex]) > 70.0:		
			MandOver70 = 1	
		if  Patient[GenderIndex] == "M" and 2012.0 - float(Patient[AgeIndex]) > 75.0:		
			MandOver75 = 1			
		if  Patient[GenderIndex] == "F" and 2012.0 - float(Patient[AgeIndex]) > 20.0:		
			FandOver20 = 1
		if  Patient[GenderIndex] == "F" and 2012.0 - float(Patient[AgeIndex]) > 25.0:		
			FandOver25 = 1
		if  Patient[GenderIndex] == "F" and 2012.0 - float(Patient[AgeIndex]) > 30.0:		
			FandOver30 = 1
		if  Patient[GenderIndex] == "F" and 2012.0 - float(Patient[AgeIndex]) > 35.0:		
			FandOver35 = 1
		if  Patient[GenderIndex] == "F" and 2012.0 - float(Patient[AgeIndex]) > 40.0:		
			FandOver40 = 1
		if  Patient[GenderIndex] == "F" and 2012.0 - float(Patient[AgeIndex]) > 45.0:		
			FandOver45 = 1
		if  Patient[GenderIndex] == "F" and 2012.0 - float(Patient[AgeIndex]) > 50.0:		
			FandOver50 = 1
		if  Patient[GenderIndex] == "F" and 2012.0 - float(Patient[AgeIndex]) > 55.0:		
			FandOver55 = 1
		if  Patient[GenderIndex] == "F" and 2012.0 - float(Patient[AgeIndex]) > 60.0:		
			FandOver60 = 1
		if  Patient[GenderIndex] == "F" and 2012.0 - float(Patient[AgeIndex]) > 65.0:		
			FandOver65 = 1		
		if  Patient[GenderIndex] == "F" and 2012.0 - float(Patient[AgeIndex]) > 70.0:		
			FandOver70 = 1	
		if  Patient[GenderIndex] == "F" and 2012.0 - float(Patient[AgeIndex]) > 75.0:		
			FandOver75 = 1	
	
	
		
		Smoking1 = 0
		Smoking2 = 0		
		Smoking3 = 0
		Smoking4 = 0
		Smoking5 = 0
		Smoking6 = 0
		
		SleepProblem = 0
		Candidiasis = 0
		
		
		BioArray = []
		BPArray = []
		
		# reset all entries...
		for SmokingStatus in SmokingStatusList:
			SmokingStatusList[SmokingStatus] = 0
	
		# reset all entries...
		for Diagnosis in DiagnosisList:
			DiagnosisList[Diagnosis] = 0

		# reset all entries...
		for Diagnosis in DiagnosisListX:
			DiagnosisListX[Diagnosis] = 0
			
		# reset all entries...
		for Medication in MedicationList:
			MedicationList[Medication] = 0
			
		# reset all entries to Average...
		for LabObservation in LabObservationList:
			LabObservationList[LabObservation] = LabObservationDefaultValueList[LabObservation]
		
		# reset all entries...
		for LabPanel in LabPanelList:
			LabPanelList[LabPanel] = 0		
			
		if (TranscriptIndex < len(SyncTranscriptSorted)):


			# Process patient transcripts
			while (Patient[0] == SyncTranscriptSorted[TranscriptIndex][1]):  # match patientGuid
			
				Transcript = SyncTranscriptSorted[TranscriptIndex]
				
				if ( Transcript[3] != "NULL" and toFloat(Transcript[4]) != 0.0 and toFloat(Transcript[5]) != 0.0 ):
					BioArray.append( [ toFloat(Transcript[3]), toFloat(Transcript[4]), toFloat(Transcript[5]) ] )
				
				#for diastolic BP under 40, skip record since a few have bad readings.
				if ( Transcript[6] != "NULL" and Transcript[7] != "NULL" and toFloat(Transcript[7]) > 40.0):
					BPArray.append( [ toFloat(Transcript[6]), toFloat(Transcript[7]) ] )
						
				TranscriptIndex = TranscriptIndex + 1			

				if (TranscriptIndex >= len(SyncTranscriptSorted)):
					break;
							
			# process Smoking records
			while (Patient[0] == SyncPatientSmokingStatusSorted[SmokingStatusIndex][1]):  # match patientGuid

				SmokingStatus = SyncPatientSmokingStatusSorted[SmokingStatusIndex]
				
				if SmokingStatus[2] in SmokingStatusList:
					SmokingStatusList[SmokingStatus[2]] = 1
			
				# special
				if SmokingStatus[2] == "02116D5A-F26C-4A48-9A11-75AC21BC4FD3" :
					Smoking1 = 1
				if SmokingStatus[2] == "02116D5A-F26C-4A48-9A11-75AC21BC4FD3" or SmokingStatus[2] == "2548BD83-03AE-4287-A578-FA170F39E32F":
					Smoking2 = 1	
				if SmokingStatus[2] == "02116D5A-F26C-4A48-9A11-75AC21BC4FD3" or SmokingStatus[2] == "2548BD83-03AE-4287-A578-FA170F39E32F" or SmokingStatus[2] == "0815F240-3DD3-43C6-8618-613CA9E41F9F":
					Smoking3 = 1
				if SmokingStatus[2] == "02116D5A-F26C-4A48-9A11-75AC21BC4FD3" or SmokingStatus[2] == "2548BD83-03AE-4287-A578-FA170F39E32F" or SmokingStatus[2] == "0815F240-3DD3-43C6-8618-613CA9E41F9F" or SmokingStatus[2] == "C12C2DB7-D31A-4514-88C0-42CBD339F764" :
					Smoking4 = 1
				if SmokingStatus[2] == "02116D5A-F26C-4A48-9A11-75AC21BC4FD3" or SmokingStatus[2] == "2548BD83-03AE-4287-A578-FA170F39E32F" or SmokingStatus[2] == "0815F240-3DD3-43C6-8618-613CA9E41F9F" or SmokingStatus[2] == "C12C2DB7-D31A-4514-88C0-42CBD339F764" or SmokingStatus[2] == "FCD437AA-0451-4D8A-9396-B6F19D8B25E8" :
					Smoking5 = 1
				if SmokingStatus[2] == "02116D5A-F26C-4A48-9A11-75AC21BC4FD3" or SmokingStatus[2] == "2548BD83-03AE-4287-A578-FA170F39E32F" or SmokingStatus[2] == "0815F240-3DD3-43C6-8618-613CA9E41F9F" or SmokingStatus[2] == "C12C2DB7-D31A-4514-88C0-42CBD339F764" or SmokingStatus[2] == "FCD437AA-0451-4D8A-9396-B6F19D8B25E8" or SmokingStatus[2] == "DD01E545-D7AF-4F00-B248-9FD40010D81D" :
					Smoking6 = 1					
			
				SmokingStatusIndex = SmokingStatusIndex + 1
				
				if (SmokingStatusIndex >= len(SyncPatientSmokingStatusSorted)):
					break;
				
			# process Diagnosis records
			while (Patient[0] == SyncDiagnosisSorted[DiagnosisIndex][1]):  # match patientGuid

				Diagnosis = SyncDiagnosisSorted[DiagnosisIndex]
				
				if Diagnosis[2] in DiagnosisList:
					DiagnosisList[Diagnosis[2]] = 1
			
				# special
				if Diagnosis[2] == "780.51" or Diagnosis[2] == "780.57" or Diagnosis[2] == "780.53":
					SleepProblem = 1
				if Diagnosis[2] == "112.1" or Diagnosis[2] == "112.8":
					Candidiasis = 1			
			
			
				DiagnosisIndex = DiagnosisIndex + 1
				
				if (DiagnosisIndex >= len(SyncDiagnosisSorted)):
					break;			
			
			
			# process Diagnosis records without decimal and to the right
			while (Patient[0] == SyncDiagnosisSorted[DiagnosisIndexX][1]):  # match patientGuid

				Diagnosis = SyncDiagnosisSorted[DiagnosisIndexX]
				#print "DIAG: ", Diagnosis[2]
				Diagnosis2 = Diagnosis[2].split(".")[0]
				if Diagnosis2 in DiagnosisListX:
					DiagnosisListX[Diagnosis2] = 1
			

				DiagnosisIndexX = DiagnosisIndexX + 1
				
				if (DiagnosisIndexX >= len(SyncDiagnosisSorted)):
					break;	
			
			
			
			# process Medication records
			while (Patient[0] == SyncMedicationSorted[MedicationIndex][1]):  # match patientGuid

				Medication = SyncMedicationSorted[MedicationIndex]
				
				if Medication[2] in MedicationList:
					MedicationList[Medication[2]] = 1
			
				MedicationIndex = MedicationIndex + 1
				
				if (MedicationIndex >= len(SyncMedicationSorted)):
					break;	

			# process Lab records
			if (LabObservationIndex < len(SyncLabObservationSorted)): # patients at end of list without lab records
				while (Patient[0] == SyncLabObservationSorted[LabObservationIndex][3]):  # match patientGuid repalced for LabPanelGuid
					#print "Patient Match", LabObservationIndex
					LabObservation = SyncLabObservationSorted[LabObservationIndex]
					
					# Sets only last value, my need to improve this........ *************
					if LabObservation[0] in LabObservationList and LabObservation[5] != "NULL" and LabObservation[5] != "":
						#print "SET", LabObservation[5], Patient[0]
						LabObservationList[LabObservation[0]] = LabObservation[5]
				
					LabObservationIndex = LabObservationIndex + 1
					#print "Patient Match", LabObservationIndex, "Next", SyncLabObservationSorted[LabObservationIndex]
					if (LabObservationIndex >= len(SyncLabObservationSorted)):
						break;	
						
			#print LabObservationList	
					
			# process LabPanel records
			if (LabPanelIndex < len(SyncLabPanelSorted)): # patients at end of list without lab records
				while (Patient[0] == SyncLabPanelSorted[LabPanelIndex][2]):  # match patientGuid

					LabPanel = SyncLabPanelSorted[LabPanelIndex]
					
					if LabPanel[0] in LabPanelList:
						LabPanelList[LabPanel[0]] = 1
						
					LabPanelIndex = LabPanelIndex + 1
					
					if (LabPanelIndex >= len(SyncLabPanelSorted)):
						break;				
					
				#print LabPanelList	
					
			# One row has no BP, hence we insert the average value.
			if ( len(BPArray) == 0 ):
				BPArray.append([126.7,76.6])
			

			BioArraySorted = sorted(BioArray, key=lambda bmi_: bmi_[2]) # sort by bmi
			BPArraySorted = sorted(BPArray, key=lambda systolic_: systolic_[0]) # sort by systolic BP
			
			# find the median value in the set
			height = BioArraySorted[ int(math.floor( float(len(BioArraySorted) ) / 2.0 )) ][0]
			weight = BioArraySorted[ int(math.floor( float(len(BioArraySorted) ) / 2.0 )) ][1]
			bmi = BioArraySorted[ int(math.floor( float(len(BioArraySorted) ) / 2.0 )) ][2]
			
			systolicBP = BPArraySorted[ int(math.floor( float(len(BPArraySorted)) / 2.0 ))][0]
			diastolicBP = BPArraySorted[ int(math.floor( float(len(BPArraySorted)) / 2.0 ))][1]
				
			if ( len(BioArray) == 0 ) :
				HeightIsNone = HeightIsNone + 1 
			if ( len(BioArray) == 0 ) :
				WeightIsNone = WeightIsNone + 1 
			if ( len(BioArray) == 0 ) :
				BMIIsNone = BMIIsNone + 1 
			if ( len(BPArray) == 0 ) :
				SystolicBPIsNone = SystolicBPIsNone + 1 
			if ( len(BPArray) == 0 ) :
				DiastolicBPIsNone = DiastolicBPIsNone + 1 
				
			
			#NOTE, all heart rates are NULL
			#and 2878 records have null for resp rate, which likely has very poor coeralation to DM
			
			
			#print [Patient[1], Patient[2], Patient[3], height, weight, bmi, systolicBP, diastolicBP]
			
			# DMIndicator, Gender, YearOfBirth, Height, Wieght, bmi, systolicBP, diastolicBP
			if ( dataSet == "training" ):
				#if ( Patient[0] == "FB12B734-8BAD-473C-8830-9739EEA01034"):
				#print Patient[1], Patient[2], Patient[3], height, weight, bmi, systolicBP, diastolicBP, SmokingStatusList, DiagnosisList, MedicationList
				#print Patient[0]

				#Output.append([Patient[1], Patient[2], Patient[3], height, weight, bmi, systolicBP, diastolicBP, copy.deepcopy(SmokingStatusList), copy.deepcopy(DiagnosisList), copy.deepcopy(MedicationList)])#, copy.deepcopy(DiagnosisList), copy.deepcopy(MedicationList)])
				
				csv_io.write_delimited_file("PreProcessData/" + dataSet + "_PreProcess.csv", [[Patient[1], Patient[2], Patient[3], height, weight, bmi, systolicBP, diastolicBP, copy.deepcopy(SmokingStatusList), copy.deepcopy(DiagnosisList), copy.deepcopy(DiagnosisListX), copy.deepcopy(MedicationList), copy.deepcopy(LabObservationList), copy.deepcopy(LabPanelList),Smoking1,Smoking2,Smoking3,Smoking4,Smoking5,Smoking6,SleepProblem,Candidiasis,MandOver20,MandOver25,MandOver30,MandOver35,MandOver40,MandOver45,MandOver50,MandOver55,MandOver60,MandOver65,MandOver70,MandOver75,FandOver20,FandOver25,FandOver30,FandOver35,FandOver40,FandOver45,FandOver50,FandOver55,FandOver60,FandOver65,FandOver70,FandOver75]], filemode="a")

			# Gender, YearOfBirth, Height, Wieght, bmi, systolicBP, diastolicBP
			if ( dataSet == "test" ) :
				#Output.append([Patient[1], Patient[2], height, weight, bmi, systolicBP, diastolicBP, copy.deepcopy(SmokingStatusList), copy.deepcopy(DiagnosisList), copy.deepcopy(MedicationList)])
				csv_io.write_delimited_file("PreProcessData/" + dataSet + "_PreProcess.csv", [[Patient[1], Patient[2], height, weight, bmi, systolicBP, diastolicBP, copy.deepcopy(SmokingStatusList), copy.deepcopy(DiagnosisList), copy.deepcopy(DiagnosisListX), copy.deepcopy(MedicationList), copy.deepcopy(LabObservationList), copy.deepcopy(LabPanelList),Smoking1,Smoking2,Smoking3,Smoking4,Smoking5,Smoking6,SleepProblem,Candidiasis,MandOver20,MandOver25,MandOver30,MandOver35,MandOver40,MandOver45,MandOver50,MandOver55,MandOver60,MandOver65,MandOver70,MandOver75,FandOver20,FandOver25,FandOver30,FandOver35,FandOver40,FandOver45,FandOver50,FandOver55,FandOver60,FandOver65,FandOver70,FandOver75]], filemode="a")

	
				
				
	print "HeightIsNone: ", HeightIsNone	
	print "WeightIsNone: ", WeightIsNone	
	print "BMIIsNone: ", BMIIsNone
	print "SystolicBPIsNone: ", SystolicBPIsNone	
	print "DiastolicBPIsNone: ", DiastolicBPIsNone	


	
	#csv_io.write_delimited_file("PreProcessData/" + dataSet + "_PreProcess.csv", Output)

	if ( dataSet == "test"):
		csv_io.write_delimited_file("PreProcessData/" + dataSet + "_PatientGuid.csv", PatientGuidList)
		
	#var = raw_input("Enter to terminate.")	
			
								
if __name__=="__main__":
	PreProcess()