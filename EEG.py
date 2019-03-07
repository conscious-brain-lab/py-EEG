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
from scipy.ndimage import measurements
import numpy as np

from subprocess import *
from pylab import *
from numpy import *
from math import *
from os import listdir

from IPython import embed as shell

import pandas as pd

import mne
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_morlet
from mne import io, EvokedArray
from mne.datasets import sample
from mne.decoding import Vectorizer, get_coef
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
from mne.viz import plot_evoked_topo

from mne.stats import spatio_temporal_cluster_test


from mne.decoding import (SlidingEstimator, GeneralizingEstimator,
                          cross_val_multiscore, LinearModel, get_coef, CSP)
from mne.filter import filter_data

class EEGsession(object):
	def __init__(self, dataDir):
		self.dataDir = dataDir
		self.chanSel = {}
		self.chanSel['OCC'] = ['Oz','O1','O2', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'Iz']
		# self.chanSel['']

	def prep(self,ID,**kwargs):
		# this method loads all necesary files and pre-defines some settings
		self.ID = ID
		self.subject,self.index,self.task  = self.ID.split('_')

		self.eeg_filename = glob.glob(os.path.join(self.dataDir, self.task + '/' , self.subject + '/',  self.subject + '_' + str(self.index) + '*.bdf'))[-1]
		if kwargs is not None:
			for argument in ['bad_chans','event_ids']:
				value = kwargs.pop(argument, 0)
				setattr(self, argument, value)

		try:
			self.epochFilename = glob.glob(os.path.join(self.dataDir, self.task + '/' ,self.subject + '/',  self.subject + '_' + str(self.index) + '_' + self.task +'_epo.fif'))[-1]	
			self.epochs =  mne.read_epochs(self.epochFilename, preload=True)
			print "epoch files found and loaded"
		except:
			self.raw =  mne.io.read_raw_edf(self.eeg_filename, eog = ['HL','HR','VU','VD'],
				misc = ['M1','M2'], preload=True)
			print "Epoch-file not found, continueing pre-processing"
			self.preproc()
	
	def preproc(self):
		""" This method runs all the necessary pre-processing steps on the raw EEG-data. 
			Included are:
			- re-referencing
			- blink detection ()
			- creating epochs 
			- ICA (+ selection and removal)
		"""
		self.raw.set_montage(mne.channels.read_montage('biosemi64'))
		self.raw.set_eeg_reference(ref_channels = ['M1','M2'], projection=False)
		self.raw.drop_channels(['ECD','ECU'])

		# Detect blinks
		eog_events = mne.preprocessing.find_eog_events(self.raw)
		n_blinks = len(eog_events)

		# Center to cover the whole blink with full duration of 0.5s:
		onset = eog_events[:, 0] / self.raw.info['sfreq'] - 0.25
		duration = np.repeat(0.5, n_blinks)
		self.raw.annotations = mne.Annotations(onset, duration, ['bad blink'] * n_blinks,
                                  orig_time=self.raw.info['meas_date'])
		# self.raw.set_annotations(annotations)
		picks_eeg = mne.pick_types(self.raw.info, meg=False, eeg=True, eog=True,
                       stim=False)
		# self.raw.		
		# self.raw.copy().drop_channels(self.raw.copy().info['bads'] )
		self.raw.info['bads'] = self.bad_chans 
		if len(self.raw.info['bads']) > 0:
			self.raw.interpolate_bads(reset_bads=True) 
		self.events = mne.find_events(self.raw)

		for ev in range(1,self.events.shape[0]): # Remove any events with weirdly short intervals (in the merged data-files, this happens at the "zip"-location)
			if self.events[ev,0] - self.events[ev-1,0] < 50:
				self.events[ev,2] = 0

		self.epochs = mne.Epochs(self.raw, self.events, event_id=self.event_ids,
			 preload=True, tmin = -1.00, tmax = 2.0, baseline = (-0.200,0), 
			 picks=picks_eeg, reject_by_annotation=True)
		
		ica = ICA(n_components=25, method='fastica')
		ica.fit(self.epochs.copy(),decim=4)
		bad_idx, scores = ica.find_bads_eog(self.epochs, ch_name = 'VU', threshold=2)
		ica.apply(self.epochs, exclude=bad_idx)
		self.epochs.save(self.eeg_filename[:-4]+'_epo.fif')

