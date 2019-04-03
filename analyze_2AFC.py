#!/usr/bin/env python
# encoding: utf-8
"""
exp.py

Created by Stijn Nuiten on 2018-02-14.
Copyright (c) 2018 __MyCompanyName__. All rights reserved.
"""
import os, sys, datetime
from os import listdir
import subprocess, logging
import datetime, time, math
import pickle

import re

import glob

import scipy as sp
import scipy.stats as stats
import scipy.signal as signal
import numpy as np

from subprocess import *
from pylab import *
from numpy import *
from math import *
from os import listdir

from IPython import embed as shell

import pandas as pd

from EEG import * 
from MVPA import *

subjects = ['26','30']
ids = [0] 
tasks = ['loc']
baseDir = '/Users/stijnnuiten/Documents/UvA/Data/perception/'
bads = {'26_0_loc':[],'30_0_loc':[]}
event_ids = {'loc':	{"stim/present/left/vert": 3848, "stim/present/left/horz": 3849, 
						"stim/present/right/vert": 3850, "stim/present/right/horz": 3851, 
						"stim/absent": 3852, "choice/present": 3856, "choice/absent": 3857}
						}

if __name__ == '__main__':
	eeg = EEG(baseDir=baseDir)
	mvpa = MVPA(baseDir=baseDir)
	for t in tasks:
		for i in ids:
		 	# for subj in subjects:
				# ID = subj+'_'+str(i)+'_'+t
				# eeg.prep(ID=ID,bad_chans=bads[ID],event_ids=event_ids[t])
				# eeg.erp(conds = ["stim/present","stim/absent"],chan=['Oz','Pz','PO3'],lims = [-0.2,1.0])
				# eeg.TFdecomp(freqs = np.logspace(*np.log10([4, 35]), num=20), lims= [-0.2,1.0], baseline = [-0.2, 0.0], method='morlet', decim=50)
				# eeg.extractTFRevents(event_ids={'present': [3848,38409,3850,3851],'absent':[3852]})

				# mvpa.prep(ID=ID,event_ids={'stim/present': [3848,3849,3850,3851],'stim/absent':[3852]})
				# mvpa.SVM( method='tempGen',times=[-0.2, 1.0], decim=8, supra=False)

			# GROUP ANALYSIS
			# eeg.groupTF(task=t,idx=i,event_ids={'present': [3848,38409,3850,3851],'absent':[3852]},subs=subjects, chanSel='OCC')
			mvpa.groupLevel(subs=subjects, task=t, idx=i, method='tempGen', supra = False, selection = 'ALL')
