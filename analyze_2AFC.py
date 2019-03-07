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

subjects = ['27']
ids = [0] 
tasks = ['attend']
dataDir = '/Users/stijnnuiten/Documents/UvA/Data/perception/'

if __name__ == '__main__':
	eeg = EEG(dataDir=dataDir)
	for t in tasks:
		for i in ids:
			for subj in subjects:
				eeg.prep(subject=subj, index=i,  task = t,bad_chans=[])
