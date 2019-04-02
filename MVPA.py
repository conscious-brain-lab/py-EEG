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


class MVPA(object):
	def __init__(self,baseDir):
		self.baseDir = baseDir

	def prep(self, ID,event_ids, selection = 'ALL'):
		self.ID = ID
		self.subject,self.index,self.task  = self.ID.split('_')
		self.subDir = os.path.join(self.baseDir, self.task + '/' ,self.subject + '/')
		self.event_ids = event_ids
		print self.ID

		self.epochFilename = glob.glob(os.path.join(self.subDir, '*' + str(self.index) + '*_epo.fif'))[-1]
		self.epochs =  mne.read_epochs(self.epochFilename, preload=True)
		self.params = pd.read_csv(os.path.join(self.subDir, self.subject + '_' + str(self.index) + '_params.csv'))

		self.selection = selection
		self.chanSel = {}
		self.chanSel['ALL'] = self.epochs.info['ch_names'][0:64]
		self.chanSel['OCC'] = ['Oz','O1','O2', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'Iz']
		self.epochs.pick_channels(self.chanSel[selection])

		self.epochsClass1 = self.epochs[event_ids.keys()[0]]
		self.epochsClass2 = self.epochs[event_ids.keys()[1]]
 
		self.decDir = os.path.join(self.subDir + 'decoding/')
		if not os.path.isdir(self.decDir):
			os.mkdir(self.decDir)

	def SVM(self, method, times=[-0.2, 1.0], decim=8, supra=False):

		self.epochsClass1.crop(times[0],times[1])
		self.epochsClass2.crop(times[0],times[1])
		self.epochsClass1 =  self.epochsClass1.decimate(decim)
		self.epochsClass2 =  self.epochsClass2.decimate(decim)

		conds = self.event_ids.keys()
		condname = conds[0] + ' vs. ' + conds[1]
		if '/' in condname:
			condname = condname.replace('/','-')

		y = np.hstack([np.zeros(len(self.epochsClass1),dtype='bool'), np.ones(len(self.epochsClass2),dtype ='bool')])
		X = np.concatenate([self.epochsClass1.get_data(), self.epochsClass2.get_data()])

		svc = SVC(C=1, kernel='linear')
		clf = make_pipeline(StandardScaler(), svc)
		cv  = ShuffleSplit(n_splits=10, test_size=0.1)
		time_decod = SlidingEstimator(clf, n_jobs=1, scoring='roc_auc')
		time_gen = GeneralizingEstimator(clf, n_jobs=1, scoring='roc_auc')

		chance = 1.0/len(conds)

		if supra:
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

		scores = np.mean(scores, axis=0)
		scoresSEM = np.std(scoresAll,axis=0)/np.sqrt(scoresAll.shape[0])
		df = pd.DataFrame(data=scores)
		
		df.to_csv(os.path.join(self.decDir, self.subject + '_' + str(self.index) + '_SVM_' + method + '_' + condname + ' ' + self.selection + (' supra' * supra) +'.csv'))

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

			plt.savefig(fname = os.path.join(plotDir, self.subject + '_' + str(self.index) + '_tempGen ' +  condname + ' ' + self.selection + (' supra' * supra) + '.pdf'),format= 'pdf')
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
			plt.savefig(fname = os.path.join(plotDir, self.subject + '_' +  str(self.index) + '_SVM ' + condname + ' ' + self.selection + (' supra' * supra) + '.pdf'),format= 'pdf')
			plt.close()

	def modelLoc(self):
		# For now, this method trains on localizer data and tests on discrim/detect data (creating a generalization matrix). 
		# As an example, we will decode presence/absence of a grating (in discrimination: i.e. left vs right).
		# Here, the decoding analysis runs over ALL trials (correct, miss, FA, CR).
		svc = SVC(C=1, kernel='linear')

		clf = make_pipeline(StandardScaler(), svc) #LinearModel(LogisticRegression()))
		time_decod = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=1)
		self.stimEpochs = self.epochs["stim"].pick_types(eeg=True).decimate(16)			
		if np.unique(self.stimEpochs.event_id.keys()).shape[0] > 2:
			epochs =  [self.stimEpochs, self.stimEpochs['present']]
		else:
			epochs = [self.stimEpochs]

		conds = ['presence', 'orientation']
		for i in range(len(epochs)):
			# Retrieve all data
			if i == 0:
				# y = 1* (epochs[i].events[:,2]< 3850) 
				y = 1* (epochs[i].events[:,2]< max(epochs[i].events[:,2])) 

			elif i == 1:
				if int(self.subject) > 19 and self.task=='loc':
					y = 1* (epochs[i].events[:,2]< 3850) & (epochs[i].events[:,2] != 3852) 
				else:
					y = 1* (epochs[i].events[:,2] < 3849)

		# Drop all channels but EEG and downsample to 128Hz
		if self.task == 'detect':
			locEpochs = self.locEpochs["left","right","absent"].pick_types(eeg=True).decimate(16)
		elif self.task == 'discrim':
			locEpochs = self.locEpochs["left","right"].pick_types(eeg=True).decimate(16)

		time_decod.fit(X=locEpochs.get_data(),
			y= locEpochs.events[:,2]<max(locEpochs.events[:,2])) #== 3848
		#1* (locEpochs.events[:,2]< 3850) & (locEpochs.events[:,2] != 3852) )
		scores = time_decod.score(X=epochs.get_data(),
						y=epochs.events[:,2]<max(epochs.events[:,2])  ) #== 3848

		fig, ax = plt.subplots(1)
		im = ax.matshow(scores, vmin=0.4, vmax=0.6, cmap='RdBu_r', origin='lower',
		                extent=epochs.times[[0, -1, 0, -1]])
		ax.axhline(0., color='k')
		ax.axvline(0., color='k')
		ax.xaxis.set_ticks_position('bottom')
		ax.set_xlabel('Testing Time (s)')
		ax.set_ylabel('Training Time (s)')
		ax.set_title('Generalization across time and condition')
		plt.colorbar(im, ax=ax)
		plt.show()
		# shell()
		df = pd.DataFrame(data=scores)
		df.to_csv(os.path.join(self.baseDir,self.task + '/' , self.subject + '/' , self.subject + '_' + str(self.index) +  '_loc_accuracies.csv'))
		plt.savefig(fname = self.baseDir + 'figs/' +self.task + '/decoding/indiv/' + self.subject + '/' + self.subject + '_' + self.task + '_' +  str(self.index) + '_loc_crossdecode.pdf',format= 'pdf')
			
		scoresDiag = [scores[i][i] for i in range(len(scores))] 
		# # Plot
		fig, ax = plt.subplots()
		ax.plot(epochs.times, scoresDiag, label='score')
		# ax.fill_between(epochs.times, scores-scoresSEM,scores+scoresSEM,alpha=0.2)
		ax.axhline(.5, color='k', linestyle='--', label='chance')
		ax.set_xlabel('Times')
		ax.set_ylabel('AUC')  # Area Under the Curve
		ax.legend()
		ax.axvline(.0, color='k', linestyle='-')
		ax.set_title('Sensor space decoding ' + self.task)
		plt.savefig(fname = self.baseDir + 'figs/' +self.task + '/decoding/indiv/' + self.subject + '/' + self.subject + '_' + self.task + '_loc_crossdecodeDiag.pdf',format= 'pdf')

	def groupLevel(self, subjects, ids, task, supra = '', selection = 'ALL'):	
		self.supra = supra
		self.task = task
		self.conds = ['response','presence','orientation'] # 
		self.ids = ids

		for c in range(len(self.conds)):
			Mat = np.array([])
			for s in subjects:
				classDir = self.baseDir + self.task + '/' + s + '/decoding/'
				dat = pd.read_csv(glob.glob(os.path.join(classDir, s+ '_' + str(self.ids) + '_SVM_' + self.conds[c] + '*' + selection + '_' + self.supra + '_*.csv'))[-1])
				Mat = np.append(Mat,dat.values[:,1])	

			allDat = Mat.reshape(len(subjects),dat.shape[0])

			allDatMean = allDat.mean(axis=0)
			allDatSEM = np.std(allDat,axis=0)/np.sqrt(allDat.shape[0])
			pvals, pvalsGN = self.prevInference(data=allDat)

			times = linspace(-200,1200,allDat.shape[1])
			plt.suptitle('Mean decoding performance (N=%i). %s %s' %(allDat.shape[0], selection, supra))
			plt.subplot(len(self.conds),1,c+1)
			h1 = plt.plot(times,allDatMean)
			h2 = plt.fill_between(times, allDatMean-allDatSEM,allDatMean+allDatSEM,alpha=0.5)

			# plt.xticks(np.arange(-200,1400,200))
			# if not self.conds[c] == 'orientation':
			plt.ylim(0.4,1.1)
			plt.yticks(np.arange(0.40,1.1, 0.10))
				
			plt.axhline(0.5, color='k',linestyle='-.')
			plt.axvline(0., color='k')
			# h3, = plt.plot(linspace(-200,1200,allDat.shape[1]),(pvals<0.05)*0.45 , 'ro', label='Majority Null')
			h4, = plt.plot(linspace(-200,1200,allDat.shape[1]),(pvalsGN<0.05)*0.45 , 'go', label='Global Null')
			plt.legend(handles=[ h4], labels=['Global Null'])
			plt.title(self.conds[c])

		plt.subplots_adjust(hspace = 0.5)
		plt.savefig(fname = self.baseDir + 'figs/'  + self.task + '/decoding/group/Decoding' + str(self.ids) + self.supra + '_' + selection + '.pdf',format= 'pdf')
		plt.close()
