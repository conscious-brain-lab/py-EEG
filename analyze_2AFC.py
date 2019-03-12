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

from EEG import EEGsession as EEG

subjects = ['30']
ids = [0] 
tasks = ['loc']
dataDir = '/Users/stijnnuiten/Documents/UvA/Data/perception/'
bads = {'30_0_loc':[]}
event_ids = {'loc':	{"stim/present/left/vert": 3848, "stim/present/left/horz": 3849, 
						"stim/present/right/vert": 3850, "stim/present/right/horz": 3851, 
						"stim/absent": 3852, "choice/present": 3856, "choice/absent": 3857}
						}

if __name__ == '__main__':
	eeg = EEG(dataDir=dataDir)
	for t in tasks:
		for i in ids:
			for subj in subjects:
				ID = subj+'_'+str(i)+'_'+t
				eeg.prep(ID=ID,bad_chans=bads[ID],event_ids=event_ids[t])
				eeg.erp(conds = ["stim/present","stim/absent"])
