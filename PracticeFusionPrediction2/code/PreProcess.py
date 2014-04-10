#!/opt/local/bin python2.7

import csv_io
import math
from math import log
import string
import copy
import operator
import os

def toFloat(str):
	return float(str)

def PreProcess():
	PreProcessRun("training")
	PreProcessRun("test")	
	
def PreProcessRun(dataSet):
	print
	print "DataSet: ", dataSet
	
	DataClassList = ["Gender", "Age", "height", "weight", "bmi", "systolicBP", "diastolicBP"]
	
	# need to delete file, since we are doing file appends at the bottom (problems holding all data in memory)
	if ( os.path.exists("PreProcessData/" + dataSet + "_PreProcess.csv") ):
		os.remove("PreProcessData/" + dataSet + "_PreProcess.csv")

	SyncPatient = csv_io.read_data("../" + dataSet + "Set/" + dataSet + "_SyncPatient.csv")
	SyncPatientSorted = sorted(SyncPatient, key=lambda patient: patient[0])

	
	SyncTranscript = csv_io.read_data("../" + dataSet + "Set/" + dataSet + "_SyncTranscript.csv")
	SyncTranscriptSorted = sorted(SyncTranscript, key=lambda patient: patient[1])


	
	SyncPatientSmokingStatus = csv_io.read_data("../" + dataSet + "Set/" + dataSet + "_SyncPatientSmokingStatus.csv")
	SyncPatientSmokingStatusSorted = sorted(SyncPatientSmokingStatus, key=lambda patient: patient[1])

	SmokingStatusList = {}
	for Record in SyncPatientSmokingStatusSorted:
		if Record[2] not in SmokingStatusList:
			SmokingStatusList[Record[2]] = 0;
			
	
	DiagnosisList = {}
	SyncDiagnosis = csv_io.read_data("../testSet/test_SyncDiagnosis.csv")
	for Record in SyncDiagnosis:
		if Record[2] not in DiagnosisList:
			DiagnosisList[Record[2]] = 0;
		else:
			DiagnosisList[Record[2]] += 1
	SyncDiagnosis = csv_io.read_data("../trainingSet/training_SyncDiagnosis.csv")
	for Record in SyncDiagnosis:
		if Record[2] not in DiagnosisList:
			DiagnosisList[Record[2]] = 0;
		else:
			DiagnosisList[Record[2]] += 1
	

	# remove from diagnosis list items occuring less than 30 times...
	for key in DiagnosisList.keys():
		if DiagnosisList[key] < 30:
			del DiagnosisList[key]
			

	
	SyncDiagnosis = csv_io.read_data("../" + dataSet + "Set/" + dataSet + "_SyncDiagnosis.csv")
	SyncDiagnosisSorted = sorted(SyncDiagnosis, key=lambda patient: patient[1])		


	# diagnosis with decimal digits removed.
	DiagnosisListX = {}
	SyncDiagnosis = csv_io.read_data("../testSet/test_SyncDiagnosis.csv")
	for Record in SyncDiagnosis:
		Record2 = Record[2].split(".")[0]
		if Record2 not in DiagnosisListX:
			DiagnosisListX[Record2] = 0;
		else:
			DiagnosisListX[Record2] += 1
	SyncDiagnosis = csv_io.read_data("../trainingSet/training_SyncDiagnosis.csv")
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
			

	
		
	MedicationList = {}
	SyncMedication = csv_io.read_data("../testSet/test_SyncMedication.csv")
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

	SyncMedication = csv_io.read_data("../trainingSet/training_SyncMedication.csv")
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

	

	# remove from Medication list items occuring less than 50 times...
	for key in MedicationList.keys():
		if MedicationList[key] < 30:
			del MedicationList[key]
			

			
	SyncMedication = csv_io.read_data("../" + dataSet + "Set/" + dataSet + "_SyncMedication.csv")
	SyncMedicationSorted = sorted(SyncMedication, key=lambda patient: patient[1])		
	
	
	LabObservationList = {}
	LabObservationDefaultValueList = {}
	SyncLabObservation = csv_io.read_data("../trainingSet/training_SyncLabObservation.csv", True, '","')
	#NOTE, if Record[0] is switched to "1", then a change needs to also occur in the lower section where these are processed.
	for Record in SyncLabObservation:
		if Record[0] != "NULL" and Record[5] != "NULL" and Record[5] != "":
			if Record[0] not in LabObservationList:
				LabObservationList[Record[0]] = 1
				LabObservationDefaultValueList[Record[0]] = float(Record[5])
			else:
				LabObservationList[Record[0]] += 1
				LabObservationDefaultValueList[Record[0]] += float(Record[5])
				
	# generate average values to fill in nulls....
	for LO in LabObservationList:
		LabObservationDefaultValueList[LO] = float(LabObservationDefaultValueList[LO])/float(LabObservationList[LO]);
	

	SyncLabObservation = csv_io.read_data("../" + dataSet + "Set/" + dataSet + "_SyncLabObservation.csv", True, '","')
	SyncLabPanel = csv_io.read_data("../" + dataSet + "Set/" + dataSet + "_SyncLabPanel.csv", True, '","')	

	
	LabPanelList = {}
	for Record in SyncLabPanel:
		if Record[1] not in LabPanelList: # need to split because file format has strings and ints
			LabPanelList[Record[1]] = Record[2].split(",")[0]; # map from Lab Panel to LabResult 

	
	for Record in SyncLabObservation:
		Record[3] = LabPanelList[Record[3]] # substitute LabResultGuid for LabPanelGuid
		
	SyncLabResult = csv_io.read_data("../" + dataSet + "Set/" + dataSet + "_SyncLabResult.csv", '","')	
	LabResultList = {}
	for Record in SyncLabResult:
		if Record[0] not in LabResultList: # need to split because file format has strings and ints
			LabResultList[Record[0]] = Record[2]; # map from Lab Result to Patient GUID	

	missing = 0
	found = 0
	SyncLabObservationNew = [] # must skip the unlinked records.
	for Record in SyncLabObservation:
		if Record[3] not in LabResultList:
			missing += 1
		else:
			found += 1
			Record[3] = LabResultList[Record[3]] # substitute Patient for LabResultGuid 
			SyncLabObservationNew.append(Record)
	print "Lab Panel Join: Found", found, "Missing" ,missing
	
	
	SyncLabObservationSorted = sorted(SyncLabObservationNew, key=lambda patient: patient[3])	# sort by column 3 which is not Patient GUID.	
		
		
		
	LabPanelList = {}
	SyncLabPanel = csv_io.read_data("../testSet/test_SyncLabPanel.csv")
	for Record in SyncLabPanel:
		if Record[0] not in LabPanelList:
			LabPanelList[Record[0]] = 0;
		else:
			LabPanelList[Record[0]] += 1
	SyncLabPanel = csv_io.read_data("../trainingSet/training_SyncLabPanel.csv")
	for Record in SyncLabPanel:
		if Record[0] not in LabPanelList:
			LabPanelList[Record[0]] = 0;
		else:
			LabPanelList[Record[0]] += 1	
		

	SyncLabPanel = csv_io.read_data("../" + dataSet + "Set/" + dataSet + "_SyncLabPanel.csv", True, '","')	
		
	missing = 0
	found = 0
	SyncLabPanelNew = [] # must skip the unlinked records.
	for Record in SyncLabPanel:
		if Record[2].split(",")[0] not in LabResultList:
			missing += 1
		else:
			found += 1
			Record[2] = LabResultList[Record[2].split(",")[0]] # substitute Patient for LabResultGuid 
			SyncLabPanelNew.append(Record)
	print "Lab Result Join: Found", found, "Missing" ,missing	
		
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
	csv_io.write_delimited_file("PreProcessData/DataClassList.csv", DataClassList)

	
	
	
	for Patient in SyncPatientSorted:
			
		PatientGuidList.append(Patient[0]) # used only for test data.	
			

		GenderIndex = 2
		AgeIndex = 3
		if ( dataSet == "test" ):
			GenderIndex = 1
			AgeIndex = 2
			
		
		
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
			
				SmokingStatusIndex = SmokingStatusIndex + 1
				
				if (SmokingStatusIndex >= len(SyncPatientSmokingStatusSorted)):
					break;
				
			# process Diagnosis records
			while (Patient[0] == SyncDiagnosisSorted[DiagnosisIndex][1]):  # match patientGuid

				Diagnosis = SyncDiagnosisSorted[DiagnosisIndex]
				
				if Diagnosis[2] in DiagnosisList:
					DiagnosisList[Diagnosis[2]] = 1
			
				DiagnosisIndex = DiagnosisIndex + 1
				
				if (DiagnosisIndex >= len(SyncDiagnosisSorted)):
					break;			
			
			
			# process Diagnosis records without decimal and to the right
			while (Patient[0] == SyncDiagnosisSorted[DiagnosisIndexX][1]):  # match patientGuid

				Diagnosis = SyncDiagnosisSorted[DiagnosisIndexX]

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
	
					LabObservation = SyncLabObservationSorted[LabObservationIndex]
					
					# Sets only last value, my need to improve this........ *************
					if LabObservation[0] in LabObservationList and LabObservation[5] != "NULL" and LabObservation[5] != "":
						LabObservationList[LabObservation[0]] = LabObservation[5]
				
					LabObservationIndex = LabObservationIndex + 1
					if (LabObservationIndex >= len(SyncLabObservationSorted)):
						break;	
						
					
			# process LabPanel records
			if (LabPanelIndex < len(SyncLabPanelSorted)): # patients at end of list without lab records
				while (Patient[0] == SyncLabPanelSorted[LabPanelIndex][2]):  # match patientGuid

					LabPanel = SyncLabPanelSorted[LabPanelIndex]
					
					if LabPanel[0] in LabPanelList:
						LabPanelList[LabPanel[0]] = 1
						
					LabPanelIndex = LabPanelIndex + 1
					
					if (LabPanelIndex >= len(SyncLabPanelSorted)):
						break;				
					
					
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
				csv_io.write_delimited_file("PreProcessData/" + dataSet + "_PreProcess.csv", [[Patient[1], Patient[2], Patient[3], height, weight, bmi, systolicBP, diastolicBP, copy.deepcopy(SmokingStatusList), copy.deepcopy(DiagnosisList), copy.deepcopy(DiagnosisListX), copy.deepcopy(MedicationList), copy.deepcopy(LabObservationList), copy.deepcopy(LabPanelList)]], filemode="a")

			# Gender, YearOfBirth, Height, Wieght, bmi, systolicBP, diastolicBP
			if ( dataSet == "test" ) :
				csv_io.write_delimited_file("PreProcessData/" + dataSet + "_PreProcess.csv", [[Patient[1], Patient[2], height, weight, bmi, systolicBP, diastolicBP, copy.deepcopy(SmokingStatusList), copy.deepcopy(DiagnosisList), copy.deepcopy(DiagnosisListX), copy.deepcopy(MedicationList), copy.deepcopy(LabObservationList), copy.deepcopy(LabPanelList)]], filemode="a")

	
				
				
	print "HeightIsNone: ", HeightIsNone	
	print "WeightIsNone: ", WeightIsNone	
	print "BMIIsNone: ", BMIIsNone
	print "SystolicBPIsNone: ", SystolicBPIsNone	
	print "DiastolicBPIsNone: ", DiastolicBPIsNone	



	if ( dataSet == "test"):
		csv_io.write_delimited_file("PreProcessData/" + dataSet + "_PatientGuid.csv", PatientGuidList)
		

			
								
if __name__=="__main__":
	PreProcess()