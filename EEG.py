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
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
from mne.viz import plot_evoked_topo

from mne.stats import spatio_temporal_cluster_test


class EEGsession(object):
	def __init__(self, dataDir):
		self.dataDir = dataDir
		self.chanSel = {}
		self.chanSel['OCC'] = ['Oz','O1','O2', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'Iz']

	def prep(self,ID,preproc=False,**kwargs):
		# this method loads all necesary files and pre-defines some settings
		self.ID = ID
		self.subject,self.index,self.task  = self.ID.split('_')

		self.plotDir =  os.path.join(self.dataDir,'figs',self.task,'eeg/indiv',self.subject) 
		if not os.path.isdir(self.plotDir):
			os.makedirs(self.plotDir)	
		self.eeg_filename = glob.glob(os.path.join(self.dataDir, self.task + '/' , self.subject + '/',  self.subject + '_' + str(self.index) + '*.bdf'))[-1]
		if kwargs is not None:
			for argument in ['bad_chans','event_ids']:
				value = kwargs.pop(argument, 0)
				setattr(self, argument, value)
		if preproc:
			self.preproc()
		else:			
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

	def erp(self,conds,**kwargs):
		self.conds=conds
		if kwargs.items():
			for argument in ['chan','lims']:
				value = kwargs.pop(argument, 0)
				setattr(self, argument, value)
		cond1 = self.epochs[self.conds[0]]
		cond2 = self.epochs[self.conds[1]]

		colors = 'blue', 'red'
		evokeds = [self.epochs[name].average() for name in (conds)]
		evokeds[0].comment, evokeds[1].comment = conds

		if hasattr(self,'chan'):
			for c in range(len(self.chan)):
				pick = evokeds[0].ch_names.index(self.chan[c])
				edi = {conds[0]: evokeds[0], conds[1]: evokeds[1]}
				mne.viz.plot_compare_evokeds(edi, picks=pick, colors=colors, show=False,show_legend=True)
				mne.viz.tight_layout()
				plt.savefig(fname=os.path.join(self.plotDir,conds[0].split('/')[1] + ' vs. ' + conds[1].split('/')[1] + '_' + self.chan[c] + '.pdf'),format='pdf')			# ax[2,0].set_suptitle('Condition difference')
		
		else:
			# evokeds = [self.epochs[name].average() for name in (conds)]
			evokeds[0].comment, evokeds[1].comment = conds

			colors = 'blue', 'red'
			title = conds[0] + 'vs. ' + conds[1]
			evokeds[0].detrend(order=1)
			evokeds[1].detrend(order=1)	
			evokeds.append(mne.combine_evoked(evokeds, weights=[-1,1]))
			maxes = np.array([np.max(evokeds[i].data) for i in range(len(evokeds))])
			mins = np.array([np.min(evokeds[i].data) for i in range(len(evokeds))])
			vmax = np.max([abs(maxes), abs(mins)])*1000000
			vmin = -vmax
			# plot_evoked_topo(axes=ax[0,0],evoked=evokeds, color=colors, title=title, background_color='w',show=False)
			plotkwargs = dict(ch_type='eeg', time_unit='s',show=False)
			fig,ax = plt.subplots(3,6, figsize = (6,6))
			evokeds[2].plot_topomap(vmin=vmin,vmax=vmax,axes=ax[2,:ax.shape[1]-1],times='peaks',colorbar=True,**plotkwargs)
			peaks = [float(str(ax[2][i].title)[-9:-4]) for i in range(ax.shape[1]-1)]
			h2=evokeds[0].plot_topomap(vmin=vmin,vmax=vmax,axes=ax[0,:ax.shape[1]-1],times=peaks,colorbar=False,**plotkwargs)
			h3=evokeds[1].plot_topomap(vmin=vmin,vmax=vmax,axes=ax[1,:ax.shape[1]-1],times=peaks,colorbar=False,**plotkwargs)
			ax[2,0].set_ylabel('difference',fontsize=14,fontweight='bold')
			ax[0,0].set_ylabel(self.conds[0],fontsize=14,fontweight='bold')
			ax[1,0].set_ylabel(self.conds[1],fontsize=14,fontweight='bold')
			matplotlib.pyplot.subplots_adjust(left=0.05,right=0.9)
			plt.savefig(fname=os.path.join(self.plotDir,conds[0].split('/')[1] + ' vs. ' + conds[1].split('/')[1] + '.pdf'),format='pdf')			# ax[2,0].set_suptitle('Condition difference')

	def TFdecomp(self,method,freqs,**kwargs):
		# For now only does Morlet-wavelet + multitaper decomposition

		# extract possible arguments
		if kwargs.items():
			for argument in ['baseline_lim','baseline_method','lims','fft','itc','average']:
				value = kwargs.pop(argument, False)
				setattr(self, argument, value)
			for argument in ['decim','bandwidth']:
				value = kwargs.pop(argument, 1)
				setattr(self, argument, value)				
				
		# first create h5-filename for time-frequency data
		tf_filename = self.epochFilename.split('/')[-1][:-4] + '-tfr.h5'

		# Number of cycles dependent on frequency-band
		n_cycles = freqs/2.

		# Run tf-decomposition
		if method == 'morlet':
			self.tf = tfr_morlet(self.epochs, freqs, n_cycles = n_cycles,decim=self.decim, use_fft=self.fft, return_itc=self.itc, average = self.average)
		elif method == 'multitaper':
			self.bandwidth = self.bandwidth if self.bandwidth > 2 else 4
			self.tf = tfr_multitaper(self.epochs, freqs, time_bandwith=self.bandwidth, n_cycles = n_cycles,decim=self.decim, use_fft=self.fft, return_itc=self.itc, average = self.average)

		# baseline if necesarry
		if self.baseline_lim:
			self.tf = self.tf.apply_baseline(mode=self.baseline_method, baseline=self.baseline_lim)

		# Crop if necessary
		if self.lims:
			self.tf.crop(tmin=self.lims[0],tmax=self.lims[1])

		# Save tfr-epoch file
		self.tf.save('/'+'/'.join(self.epochFilename.split('/')[1:-1])+'/'+tf_filename, overwrite=True)




