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

import mne
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_morlet
from mne import io, EvokedArray
from mne.datasets import sample
from mne.decoding import Vectorizer, get_coef
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
from mne.viz import plot_evoked_topo

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC  
from sklearn.discriminant_analysis import _cov

from mne.decoding import (SlidingEstimator, GeneralizingEstimator,
                          cross_val_multiscore, LinearModel, get_coef, CSP)
from mne.filter import filter_data
from functions.statsfuncs import *


class MVPA(object):
	def __init__(self,baseDir):
		self.baseDir = baseDir

	def prep(self, ID,event_ids, selection = 'ALL'):
		self.ID = ID
		self.subject,self.index,self.task  = self.ID.split('_')
		self.subDir = os.path.join(self.baseDir, self.task + '/' ,self.subject + '/')
		self.event_ids = event_ids
		print self.ID

		# find epoch file and load epochs
		self.epochFilename = glob.glob(os.path.join(self.subDir, '*' + str(self.index) + '*_epo.fif'))[-1]
		self.epochs =  mne.read_epochs(self.epochFilename, preload=True)

		# load trial-parameter file
		self.params = pd.read_csv(os.path.join(self.subDir, self.subject + '_' + str(self.index) + '_params.csv'))

		# Channel selection pre-sets
		self.chanSel = {}
		self.chanSel['ALL'] = self.epochs.info['ch_names'][0:64]
		self.chanSel['OCC'] = ['Oz','O1','O2', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'Iz']
		self.chanSel['PAR'] = ['P1', 'P3', 'P5', 'P7', 'Pz', 'P2', 'P4', 'P6', 'P8']
		self.chanSel['FRO'] = ['Fp1', 'AF7', 'AF3', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz']
		self.chanSel['TMP'] = ['FT7', 'C5', 'T7', 'TP7', 'CP5', 'FT8', 'C6', 'T8', 'TP8', 'CP6']
		self.chanSel['OPA'] = ['P1', 'P3', 'P5', 'P7', 'P9', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO7', 'PO3'
		  					   'O1', 'Iz', 'Oz', 'POz', 'PO8', 'PO4', 'O2', 'PO9', 'PO10' ]
		self.chanSel['CDA'] = ['P5', 'P6', 'P7', 'P8', 'PO7', 'PO8', 'O1', 'O2', 'PO9', 'PO10'];

		# Pick the desired channels
		self.epochs.pick_channels(self.chanSel[selection])

		# split up all epochs according to stimulus class
		self.epochsClass1 = self.epochs[event_ids.keys()[0]]
		self.epochsClass2 = self.epochs[event_ids.keys()[1]]
 
 		# create output directory, if necessaru
		self.decDir = os.path.join(self.subDir + 'decoding/')
		if not os.path.isdir(self.decDir):
			os.mkdir(self.decDir)

	def SVM(self, method, times=[-0.2, 1.0], decim=8, supra=False):
		# This method trains a support vector machine on 90% of the trials to separate conditions 
		# (2 as of now), and then tests this model on the remaining 10%. This is done in a 10-fold procedure.
		# Right now, only this procedure -and calculating auc values- are included.

		# crop data if desired
		self.epochsClass1.crop(times[0],times[1])
		self.epochsClass2.crop(times[0],times[1])

		# decimate the signal to speed up, if desired
		self.epochsClass1 =  self.epochsClass1.decimate(decim)
		self.epochsClass2 =  self.epochsClass2.decimate(decim)

		# determine condition names
		conds = self.event_ids.keys()
		condname = conds[0] + ' vs ' + conds[1]
		if '/' in condname:
			condname = condname.replace('/','-') # replace '/' with '-' to prevent creating subfolders during saving of data/figures

		# give epochs labels (0: cond1, 1: cond2)
		y = np.hstack([np.zeros(len(self.epochsClass1),dtype='bool'), np.ones(len(self.epochsClass2),dtype ='bool')])
		# concatenate data belonging to both conds
		X = np.concatenate([self.epochsClass1.get_data(), self.epochsClass2.get_data()])

		# initiate the SVM
		svc = SVC(C=1, kernel='linear')
		clf = make_pipeline(StandardScaler(), svc)
		cv  = ShuffleSplit(n_splits=10, test_size=0.1)

		# time_decod is for diagonal decoding, time_gen for temporal generalization (incl. off-diagonal)
		time_decod = SlidingEstimator(clf, n_jobs=1, scoring='roc_auc')
		time_gen = GeneralizingEstimator(clf, n_jobs=1, scoring='roc_auc')

		# Determine chance level
		chance = 1.0/len(conds) 

		if supra: # average over X-trials (now only 4), to reduce noise and increase decoding accuracy, if desired
			# Find data belonging to present vs absent
			Class1TrialDat = X[~y,:,:]
			Class2TrialDat  = X[y,:,:]

			# Create arrays of random trial orders
			Class1ShuffledOrder = np.argsort(np.random.rand(len(Class1TrialDat)))
			Class2ShuffledOrder  = np.argsort(np.random.rand(len(Class2TrialDat)))

			# loop over random order, in groups of 4, and average the data over those trials
			Class1SupraDat = np.array([mean(Class1TrialDat[[Class1ShuffledOrder[4*j:4*(j+1)]],:,:], axis=1 ) for j in range(Class1ShuffledOrder.shape[0]/4)]).squeeze()
			Class2SupraDat  = np.array([mean(Class2TrialDat[[Class2ShuffledOrder[4*k:4*(k+1)]],:,:], axis=1 ) for k in range(Class2ShuffledOrder.shape[0]/4)]).squeeze()

			# now 
			y = np.concatenate((np.ones(Class1SupraDat.shape[0]), np.zeros(Class2SupraDat.shape[0])), axis=0)
			X = np.concatenate((Class1SupraDat, Class2SupraDat), axis=0)

		if method == 'tempGen':
			scores = cross_val_multiscore(time_gen, X, y, cv=cv, n_jobs=1)
		elif method == 'diag':	
			scores = cross_val_multiscore(time_decod, X, y, cv=cv, n_jobs=1)
	
		scoresAll = scores

		# Calculate mean and SEM for plotting
		scores = np.mean(scores, axis=0)
		scoresSEM = np.std(scoresAll,axis=0)/np.sqrt(scoresAll.shape[0])

		# store averaged scores in dataFrame and save
		df = pd.DataFrame(data=scores)
		df.insert(0,'times',self.epochsClass1.times) 
		df.to_csv(os.path.join(self.decDir, self.subject + '_' + str(self.index) + '_SVM_' + method + '_' + condname + '_' + self.selection + (' supra' * supra) +'.csv'),index=False)

		plotDir = os.path.join(self.baseDir, 'figs', self.task, 'decoding/indiv',self.subject)
		if not os.path.isdir(plotDir):
			os.mkdir(plotDir)

		# Plot
		fig, ax = plt.subplots()
		if method == 'tempGen':
			im = ax.imshow(scores, interpolation='lanczos', origin='lower', cmap='RdBu_r',
			               extent=self.epochsClass2.times[[0, -1, 0, -1]], vmin=0, vmax=1.0)
			ax.set_xlabel('Testing Time (s)')
			ax.set_ylabel('Training Time (s)')
			ax.set_title('Temporal generalization' + self.task)
			ax.axvline(0, color='k')
			ax.axhline(0, color='k')
			plt.colorbar(im, ax=ax)

			plt.savefig(fname = os.path.join(plotDir, self.subject + '_' + str(self.index) + '_SVM_' + method +  condname + '_' + self.selection + (' supra' * supra) + '.pdf'),format= 'pdf')
			plt.close()

		elif method == 'diag':		
			ax.plot(self.epochsClass2.times, scores, label='score')
			ax.fill_between(self.epochsClass2.times, scores-scoresSEM,scores+scoresSEM,alpha=0.2)
			ax.axhline(chance, color='k', linestyle='-', label='chance')

			ax.set_xlabel('Times (s)')
			ax.set_ylabel('AUC')  # Area Under the Curve
			ax.legend()
			ax.axvline(0., color='k', linestyle='--')
			ax.set_title('Sensor space decoding ' + condname + ' ' + self.task)
			plt.savefig(fname = os.path.join(plotDir, self.subject + '_' +  str(self.index) + '_SVM_' + method + '_'+ condname + '_' + self.selection + (' supra' * supra) + '.pdf'),format= 'pdf')
			plt.close()

	def groupLevel(self, subs, idx, task, method, cond ='*', supra = False, selection = 'ALL',**kwargs):	
		
		# Load in data
		Mat = np.array([])
		for s in subs:
			classDir = self.baseDir + task + '/' + s + '/decoding/'
			filename = glob.glob(os.path.join(classDir, s + '_' + str(idx) + '_SVM_' + method + '_' + cond + '_' + selection + (' supra' * supra) + '.csv'))[-1]
			dat = pd.read_csv(filename)
			times = np.array(dat['times'])

			if method == 'diag':
				Mat = np.append(Mat,dat.values[:,1])	
			elif method == 'tempGen':
				Mat = np.append(Mat,dat.values[:,1:])	

		# reshape to get subject x data (time/timextime) array
		if method == 'diag':
			allDat = Mat.reshape(len(subs),dat.shape[0])
		elif method == 'tempGen':
			allDat = Mat.reshape(len(subs),dat.shape[0],dat.shape[0])

		cond = filename.split('/')[-1].split('_')[-2] if cond == '*' else cond

		chance = np.ones(allDat.shape)*0.5

		# calculate mean and SEM over subjects
		allDatMean = allDat.mean(axis=0)
		allDatSEM = np.std(allDat,axis=0)/np.sqrt(allDat.shape[0])
		
		# find index where time = 0
		tzero = (np.abs(0-times)).argmin()
		timelabels = list(linspace(tzero,len(times)-1,5,dtype=int))

		if method == 'diag':
			# calculate cluster-corrected pvals, under the majority null-hypothesis (see prevInference function)
			# And the global null hypothesis (which is theoretically problematic)
			pvals, pvalsGN = prevInference(data=allDat)

			plt.suptitle('Mean decoding performance (N=%i). %s %s' %(allDat.shape[0], selection + ' channels', ' supra' * supra ))
			h1 = plt.plot(times,allDatMean)
			h2 = plt.fill_between(times, allDatMean-allDatSEM,allDatMean+allDatSEM,alpha=0.5)
			plt.ylim(0.4,1.1)
			plt.yticks(np.arange(0.40,1.1, 0.10))
				
			plt.axhline(0.5, color='k',linestyle='-.')
			plt.axvline(0., color='k')

			# h3 = plt.plot(linspace(-200,1200,allDat.shape[1]),(pvals<0.05)*0.45 , 'ro', label='Majority Null')

			h4 = plt.plot(linspace(times[0],times[-1],allDat.shape[1]),(pvalsGN<0.05)*0.45 , 'go', label='Global Null')
			plt.legend(handles=[ h4], labels=['Global Null'])

		elif method == 'tempGen':
			# run cluster-corrected t-test (global null hypothesis only)
			t_thresh = cluster_ttest(chance,allDat,1000, 0.05)
			
			x = np.linspace(0,t_thresh.shape[1], t_thresh.shape[1]*100)
			y = np.linspace(0,t_thresh.shape[0], t_thresh.shape[0]*100)
			plt.imshow(allDatMean,cmap = 'RdBu_r',origin = 'lower',vmin=0, vmax = 1)
			cbar = colorbar()
			cbar.set_ticks(arange(0,1.1,0.25))
			cbar.ax.set_ylabel('AUC',fontsize=14,fontweight='bold',rotation='270',labelpad=20)
			for l in cbar.ax.yaxis.get_ticklabels():
   				l.set_weight("bold")
   				l.set_size(12)

			plt.contour(t_thresh,[0.5],colors='black',linestyles='solid',linewidths=[2],levels=1,origin='lower',extent=[0-0.5, x[:-1].max()-0.5,0-0.5, y[:-1].max()-0.5])
			xticks(timelabels, np.around(times[timelabels],decimals=2),fontsize=12,fontweight='bold')
			yticks(timelabels, np.around(times[timelabels],decimals=2),fontsize=12,fontweight='bold')
			xlabel('Testing time (s)',fontsize=14,fontweight='bold')
			ylabel('Training time (s)',fontsize=14,fontweight='bold')

		plt.title(cond,fontsize=14,fontweight='bold')
		plt.tight_layout()

		plt.savefig(fname = self.baseDir + 'figs/'  + task + '/decoding/group/' + method + '_' + str(idx) + '_' + cond + (' supra' * supra) + '_' + selection + '.pdf',format= 'pdf')
		plt.close()
