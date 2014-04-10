#!/usr/bin/env python



import PreProcess
import PreProcess1
import PreProcess2
import PreProcess3
import PreProcess4
import PreProcess4Base

def RunPreprocess():

	print "---PreProcess"
	PreProcess.PreProcess()
	print "---PreProcess1"
	PreProcess1.PreProcess1()
	print "---PreProcess2"
	PreProcess2.PreProcess2()
	print "---PreProcess3"
	PreProcess3.PreProcess3()
	print "---PreProcess4"	
	PreProcess4.PreProcess4()
	print "---PreProcess5,40"
	PreProcess4.PreProcess5(40)
	print "---PreProcess5,30"
	PreProcess4.PreProcess5(30)
	print "---PreProcess5Base,40"
	PreProcess4Base.PreProcess5Base(40)
	print "---PreProcess5Base,30"	
	PreProcess4Base.PreProcess5Base(30)
	
if __name__=="__main__":
	RunPreprocess()