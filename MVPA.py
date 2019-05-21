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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC  
from sklearn.discriminant_analysis import _cov

from mne.decoding import (SlidingEstimator, GeneralizingEstimator,
                          cross_val_multiscore, LinearModel, get_coef, CSP)
from mne.filter import filter_data
from functions.statsfuncs import *


class MVPA(object):
	def __init__(self,baseDir,ID,event_ids,baseline=None,balance=True,selection='ALL',condName='', dimension='timetime'):
		self.baseDir = baseDir
		self.ID = ID
		self.subject,self.index,self.task  = self.ID.split('_')
		self.subDir = os.path.join(self.baseDir,'Proc',self.task + '/' ,self.subject + '/')
		self.event_ids = event_ids
		self.selection = selection
		self.baseline = baseline
		self.balance = balance
		self.condName = condName
		self.dimension = dimension


		# Channel selection pre-sets
		self.chanSel = {}
		self.chanSel['ALL'] = ''
		self.chanSel['OCC'] = ['Oz','O1','O2', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'Iz']
		self.chanSel['PAR'] = ['P1', 'P3', 'P5', 'P7', 'Pz', 'P2', 'P4', 'P6', 'P8']
		self.chanSel['FRO'] = ['Fp1', 'AF7', 'AF3', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz']
		self.chanSel['TMP'] = ['FT7', 'C5', 'T7', 'TP7', 'CP5', 'FT8', 'C6', 'T8', 'TP8', 'CP6']
		self.chanSel['OPA'] = ['P1', 'P3', 'P5', 'P7', 'P9', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO7', 'PO3'
		  					   'O1', 'Iz', 'Oz', 'POz', 'PO8', 'PO4', 'O2', 'PO9', 'PO10' ]
		self.chanSel['CDA'] = ['P5', 'P6', 'P7', 'P8', 'PO7', 'PO8', 'O1', 'O2', 'PO9', 'PO10'];


		print (self.ID + ' ready to decode')
		# load trial-parameter file
		self.params = pd.read_csv(os.path.join(self.subDir, self.subject + '_' + str(self.index) + '_params.csv'))

		# find epoch file and load epochs
		if self.dimension == 'timetime':
			self.epochFilename = glob.glob(os.path.join(self.subDir, '*' + str(self.index) + '*_epo.fif'))[-1]
			self.epochs =  mne.read_epochs(self.epochFilename, preload=True)
			self.epochs.pick_channels(self.chanSel[self.selection])
			# split up all epochs according to stimulus class
			self.epochsClass1 = self.epochs[event_ids.keys()[0]]
			self.epochsClass2 = self.epochs[event_ids.keys()[1]]

			# If baseline = None, nothing will happen here
			self.epochsClass1.apply_baseline(baseline)
			self.epochsClass2.apply_baseline(baseline)
		
		if self.dimension == 'timefrequency':
			self.paramsFile = glob.glob(os.path.join(self.subDir, '*' + str(self.index) + '*_epo-tfr.csv'))[-1]
			self.params =np.array(pd.read_csv(self.paramsFile,header=None))[:,1]

			cond1, cond2 = self.event_ids.items() 
			class1ids = np.logical_or(self.params==cond1[1][0],self.params==cond1[1][1])   
			class2ids = np.logical_or(self.params==cond2[1][0],self.params==cond2[1][1])   

			self.epochFilename = glob.glob(os.path.join(self.subDir, '*' + str(self.index) + '*_epo-tfr.h5'))[-1]
			self.epochs = mne.time_frequency.read_tfrs(self.epochFilename)[0]
			if self.selection != 'ALL':
				self.epochs.pick_channels(self.chanSel[self.selection])			

			self.epochsClass1 = self.epochs[class1ids]
			self.epochsClass2 = self.epochs[class2ids]

 		# create output directory, if necessaru
		self.decDir = os.path.join(self.subDir + 'decoding/')
		if not os.path.isdir(self.decDir):
			os.makekdirs(self.decDir)

	def SVM(self, method, crossclass=False, times=[-0.2, 1.0], decim=8, supra=False):
		# This method trains a support vector machine on 90% of the trials to separate conditions 
		# (2 as of now), and then tests this model on the remaining 10%. This is done in a 10-fold procedure.
		# Right now, only this procedure -and calculating auc values- are included.
		
		nSplits = 10

		self.epochsClass1.crop(times[0],times[1])		# crop data if desired
		self.epochsClass2.crop(times[0],times[1])		# crop data if desired

		# decimate the signal to speed up, if desired
		if self.dimension == 'timetime':
			self.epochsClass1 =  self.epochsClass1.decimate(decim)
			self.epochsClass2 =  self.epochsClass2.decimate(decim)
			freqs = np.array([0])
		else:
			freqs = range(self.epochsClass1.data.shape[2])

		cond1, cond2 = self.event_ids.keys() # determine condition names
		# condname = cond1 + ' vs ' + cond2
		if '/' in self.condName:
			self.condName = self.condName.replace('/','-') # replace '/' with '-' to prevent creating subfolders during saving of data/figures

		chance = 1.0/len(self.event_ids.keys())  			# Determine chance level

		if supra: # average over X-trials (now only 4), to reduce noise and increase decoding accuracy, if desired
			# Find data belonging to present vs absent
			Class1ShuffledOrder = np.argsort(np.random.rand(len(self.epochsClass1)))
			Class2ShuffledOrder = np.argsort(np.random.rand(len(self.epochsClass2)))

		# initiate the SVM
		if method == 'svc':
			dec = SVC(C=1, kernel='linear')
		elif method =='lda':
			dec = LinearDiscriminantAnalysis()
		elif method == 'logreg':
			dec = LogisticRegression(solver='lbfgs')

		clf = make_pipeline(StandardScaler(), dec)
		cv  = ShuffleSplit(n_splits=nSplits, test_size=0.1)
		# time_decod is for diagonal decoding, time_gen for temporal generalization (incl. off-diagonal)
		time_decod = SlidingEstimator(clf, n_jobs=1, scoring='roc_auc')
		time_gen = GeneralizingEstimator(clf, n_jobs=1, scoring='roc_auc')

		scores = []
		for f in range(len(freqs)):
			# concatenate data belonging to both conds
			cond1Dat = self.epochsClass1.get_data()[:,:,:] if self.dimension == 'timetime' else self.epochsClass1.data[:,:,f,:]
			cond2Dat = self.epochsClass2.get_data()[:,:,:] if self.dimension == 'timetime' else self.epochsClass2.data[:,:,f,:]
			
			if supra:
				supTrial1Dat = np.ones((int(cond1Dat.shape[0]/4), cond1Dat.shape[1],cond1Dat.shape[2]))
				supTrial2Dat = np.ones((int(cond2Dat.shape[0]/4), cond2Dat.shape[1],cond2Dat.shape[2]))

				for j in list(range(int(Class1ShuffledOrder.shape[0]/4))):
					supTrial1Dat[j,:,:] = mean(cond1Dat[Class1ShuffledOrder[4*j:4*j+1],:,:],axis=0) #).squeeze()
					supTrial2Dat[j,:,:] = mean(cond2Dat[Class2ShuffledOrder[4*j:4*j+1],:,:],axis=0) #).squeeze()

				cond1Dat = supTrial1Dat		
				cond2Dat = supTrial2Dat


			y = np.hstack([np.zeros(len(cond1Dat),dtype='bool'), np.ones(len(cond2Dat),dtype ='bool')])
			X = np.concatenate([cond1Dat,cond2Dat])



			if crossclass:
				scores.append(cross_val_multiscore(time_gen, X, y, cv=cv, n_jobs=1))
			else:	
				scores.append(cross_val_multiscore(time_decod, X, y, cv=cv, n_jobs=1))
	
		scoresAll = scores

		# Calculate mean and SEM for plotting
		scores = np.mean(scores, axis=1)
		scoresSEM = np.squeeze(np.std(scoresAll,axis=1)/np.sqrt(nSplits))

		# store averaged scores in dataFrame and save
		df = pd.DataFrame(data=scores.transpose())
		df.insert(0,'times',self.epochsClass1.times) 
		df.to_csv(os.path.join(self.decDir, '_'.join((self.subject, str(self.index),method,self.condName  ,self.selection ,(' supra' * supra) +'.csv'))),index=False)



		plotDir = os.path.join('/'.join(self.baseDir.split('/')[:-2]), 'figs', self.task, 'decoding/indiv',self.subject)
		if not os.path.isdir(plotDir):
			os.makedirs(plotDir)

		# Plot
		fig, ax = plt.subplots()
		if crossclass or self.dimension == 'timefrequency':
			extent = self.epochsClass2.times[[0, -1, 0, -1]] if crossclass else np.hstack([self.epochsClass2.times[[0, -1]], freqs[0], freqs[-1]])
			yAx = 'Training Time (s)' if crossclass else 'Frequency (Hz)'
			tit = 'Temporal generalization ' + self.task if crossclass else 'Time-frequency decoding ' + self.condName			
			im = ax.imshow(scores, interpolation='lanczos', origin='lower', cmap='RdBu_r',
			               extent=extent, vmin=0.3, vmax=0.7,  aspect='auto')
			

			ax.set_xlabel('Testing Time (s)')
			ax.set_ylabel(yAx)
			ax.set_title(tit)
			ax.axvline(0, color='k')
			if crossclass:
				ax.axhline(0, color='k')

			plt.colorbar(im, ax=ax)

			plt.savefig(fname = os.path.join(plotDir, '_'.join((self.subject, str(self.index) ,method,  self.condName ,  self.selection ,(' supra' * supra) + '.pdf'))),format= 'pdf')
			plt.close()

		else:		
			ax.plot(self.epochsClass2.times, scores, label='score')
			ax.fill_between(self.epochsClass2.times, scores-scoresSEM,scores+scoresSEM,alpha=0.2)
			ax.axhline(chance, color='k', linestyle='-', label='chance')

			ax.set_xlabel('Times (s)')
			ax.set_ylabel('AUC')  # Area Under the Curve
			ax.legend()
			ax.axvline(0., color='k', linestyle='--')
			ax.set_title('Sensor space decoding ' + condname + ' ' + self.task)
			plt.savefig(fname = os.path.join(plotDir, '_'.join((self.subject, str(self.index) ,method,  self.condName ,  self.selection ,(' supra' * supra) + '.pdf'))),format= 'pdf')
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
