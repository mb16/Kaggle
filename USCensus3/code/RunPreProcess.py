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
	print "---PreProcess4,40"	
	PreProcess4.PreProcess4(40)
	print "---PreProcess4,30"
	PreProcess4.PreProcess4(30)
	print "---PreProcess4Base,40"
	PreProcess4Base.PreProcess4Base(40)
	print "---PreProcess4Base,30"	
	PreProcess4Base.PreProcess4Base(30)
	
if __name__=="__main__":
	RunPreprocess()