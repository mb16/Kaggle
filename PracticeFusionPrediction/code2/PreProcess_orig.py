#!/usr/bin/env python

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
		if DiagnosisList[key] < 50:
			del DiagnosisList[key]
			
	#print sorted(DiagnosisList.iteritems(), key=operator.itemgetter(1))
	#print len(DiagnosisList)
	
	SyncDiagnosis = csv_io.read_data("../" + dataSet + "Set/" + dataSet + "Set/" + dataSet + "_SyncDiagnosis.csv")
	SyncDiagnosisSorted = sorted(SyncDiagnosis, key=lambda patient: patient[1])		

		
		
	MedicationList = {}
	SyncMedication = csv_io.read_data("../testSet/testSet/test_SyncMedication.csv")
	for Record in SyncMedication:
		if Record[2] not in MedicationList:
			MedicationList[Record[2]] = 0;
		else:
			MedicationList[Record[2]] += 1
	SyncMedication = csv_io.read_data("../trainingSet/trainingSet/training_SyncMedication.csv")
	for Record in SyncMedication:
		if Record[2] not in MedicationList:
			MedicationList[Record[2]] = 0;
		else:
			MedicationList[Record[2]] += 1
	

	# remove from Medication list items occuring less than 50 times...
	for key in MedicationList.keys():
		if MedicationList[key] < 50:
			del MedicationList[key]
			
	print sorted(MedicationList.iteritems(), key=operator.itemgetter(1))
	print len(MedicationList)
	return
	SyncMedication = csv_io.read_data("../" + dataSet + "Set/" + dataSet + "Set/" + dataSet + "_SyncMedication.csv")
	SyncMedicationSorted = sorted(SyncMedication, key=lambda patient: patient[1])		
	
		
		
		
	Output = []
	
	TranscriptIndex = 0
	SmokingStatusIndex = 0
	DiagnosisIndex = 0
	MedicationIndex = 0
	
	HeightIsNone = 0
	WeightIsNone = 0
	BMIIsNone = 0
	SystolicBPIsNone = 0
	DiastolicBPIsNone = 0

	PatientGuidList = []
	
	for Patient in SyncPatientSorted:
			
		PatientGuidList.append(Patient[0]) # used only for test data.	
			
		#print Patient[0], TranscriptIndex, len(SyncTranscriptSorted)
		
		BioArray = []
		BPArray = []
		
		# reset all entries...
		for SmokingStatus in SmokingStatusList:
			SmokingStatusList[SmokingStatus] = 0
	
		# reset all entries...
		for Diagnosis in DiagnosisList:
			DiagnosisList[Diagnosis] = 0
	
		# reset all entries...
		for Medication in MedicationList:
			MedicationList[Medication] = 0
			
			
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
				
				if (TranscriptIndex >= len(SyncTranscriptSorted)):
					break;
				
			# process Diagnosis records
			while (Patient[0] == SyncDiagnosisSorted[DiagnosisIndex][1]):  # match patientGuid

				Diagnosis = SyncDiagnosisSorted[DiagnosisIndex]
				
				if Diagnosis[2] in DiagnosisList:
					DiagnosisList[Diagnosis[2]] = 1
			
				DiagnosisIndex = DiagnosisIndex + 1
				
				if (TranscriptIndex >= len(SyncDiagnosisSorted)):
					break;			
			
			
			# process Medication records
			while (Patient[0] == SyncMedicationSorted[MedicationIndex][1]):  # match patientGuid

				Medication = SyncMedicationSorted[MedicationIndex]
				
				if Medication[2] in MedicationList:
					MedicationList[Medication[2]] = 1
			
				MedicationIndex = MedicationIndex + 1
				
				if (TranscriptIndex >= len(SyncMedicationSorted)):
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
			if ( dataSet == "training" ) :
				Output.append([Patient[1], Patient[2], Patient[3], height, weight, bmi, systolicBP, diastolicBP, copy.deepcopy(SmokingStatusList), copy.deepcopy(DiagnosisList), copy.deepcopy(MedicationList)])
			# Gender, YearOfBirth, Height, Wieght, bmi, systolicBP, diastolicBP
			if ( dataSet == "test" ) :
				Output.append([Patient[1], Patient[2], height, weight, bmi, systolicBP, diastolicBP, copy.deepcopy(SmokingStatusList), copy.deepcopy(DiagnosisList), copy.deepcopy(MedicationList)])
				

	print "HeightIsNone: ", HeightIsNone	
	print "WeightIsNone: ", WeightIsNone	
	print "BMIIsNone: ", BMIIsNone
	print "SystolicBPIsNone: ", SystolicBPIsNone	
	print "DiastolicBPIsNone: ", DiastolicBPIsNone	


	
	csv_io.write_delimited_file("PreProcessData/" + dataSet + "_PreProcess.csv", Output)

	if ( dataSet == "test"):
		csv_io.write_delimited_file("PreProcessData/" + dataSet + "_PatientGuid.csv", PatientGuidList)
		
	#var = raw_input("Enter to terminate.")	
			
								
if __name__=="__main__":
	PreProcess()