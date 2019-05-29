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
import matplotlib.patches as patch
from mne.time_frequency import tfr_morlet
from mne import io, EvokedArray
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
from mne.viz import plot_evoked_topo

from mne.stats import spatio_temporal_cluster_test

from functions.statsfuncs import cluster_ttest


class EEG(object):
    def __init__(self, baseDir,ID=None,**kwargs):
        if kwargs.items():
            for argument in ['eegFilename','lims','bad_chans','event_ids']:
                value = kwargs.pop(argument, 0)
                setattr(self, argument, value)

        self.baseDir = baseDir
        self.chanSel = {}
        self.chanSel['ALL'] = None #self.epochs.info['ch_names'][0:64]
        self.chanSel['OCC'] = ['Oz','O1','O2', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'Iz']
        self.chanSel['PAR'] = ['P1', 'P3', 'P5', 'P7', 'Pz', 'P2', 'P4', 'P6', 'P8']
        self.chanSel['FRO'] = ['Fp1', 'AF7', 'AF3', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz']
        self.chanSel['TMP'] = ['FT7', 'C5', 'T7', 'TP7', 'CP5', 'FT8', 'C6', 'T8', 'TP8', 'CP6']
        self.chanSel['OPA'] = ['P1', 'P3', 'P5', 'P7', 'P9', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO7', 'PO3'
                               'O1', 'Iz', 'Oz', 'POz', 'PO8', 'PO4', 'O2', 'PO9', 'PO10' ]
        self.chanSel['CDA'] = ['P5', 'P6', 'P7', 'P8', 'PO7', 'PO8', 'O1', 'O2', 'PO9', 'PO10']

        self.ID = ID
        if self.ID != None:
            self.subject,self.index,self.task  = self.ID.split('_')

            self.procDir = os.path.join(self.baseDir, 'Proc', self.task, self.subject)
            if not os.path.isdir(self.procDir):
                os.makedirs(self.procDir)
            
            self.plotDir =  os.path.join(self.baseDir,'figs','indiv',self.subject)
            if not os.path.isdir(self.plotDir):
                os.makedirs(self.plotDir) 

            try:
                self.eegFilename = glob.glob(os.path.join(self.baseDir, 'Raw', self.task, self.subject, '*' + self.subject + '*' + str(self.index) + '*.bdf'))[-1]
                self.raw =  mne.io.read_raw_edf(self.eegFilename, eog = ['HL','HR','VU','VD'],
                    misc = ['M1','M2'], preload=True)
            except:
                print("RAW FILE NOT FOUND")
                pass

            try:
                self.epochFilename = glob.glob(os.path.join(self.baseDir, 'Proc', self.task, self.subject,  self.subject + '*' + str(self.index) + '*_epo.fif'))[-1]                    
                self.epochs =  mne.read_epochs(self.epochFilename, preload=True)
                print( "epoch files found and loaded")
            except:
                print ("\n\n\n\nEpoch-file not found, run preprocessing first\n\n\n\n")


        else:
            self.plotDir =  os.path.join(self.baseDir,'figs','group')
            if not os.path.isdir(self.plotDir):
                os.makedirs(self.plotDir) 

    
    def preproc(self, baseline=None, epochTime=(-1.0, 2.0), ica=True, reject=None, reject_by_annotation=False,overwrite=True):
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

        if reject_by_annotation:        # Detect and remove blink artefacts
            eog_events = mne.preprocessing.find_eog_events(self.raw)
            n_blinks = len(eog_events)

            # Center to cover the whole blink with full duration of 0.5s:
            onset = eog_events[:, 0] / self.raw.info['sfreq'] - 0.25
            duration = np.repeat(0.5, n_blinks)
            self.raw.annotations = mne.Annotations(onset, duration, ['bad blink'] * n_blinks,
                                      orig_time=self.raw.info['meas_date'])

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
             preload=True, tmin = epochTime[0], tmax = epochTime[1], baseline = baseline, 
             picks=picks_eeg, reject_by_annotation=reject_by_annotation)

        if ica:
            ica = ICA(n_components=25, method='fastica')
            ica.fit(self.epochs.copy(),decim=4)
            bad_idx, scores = ica.find_bads_eog(self.epochs, ch_name = 'VU', threshold=2)
            ica.apply(self.epochs, exclude=bad_idx)


        self.epochFilename = os.path.join(self.baseDir, 'Proc', self.task, self.subject, self.subject + '_' + str(self.index) + '_epo.fif')
        self.epochs.save(self.epochFilename,overwrite=overwrite)

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
                plt.savefig(fname=os.path.join(self.plotDir,conds[0].split('/')[1] + ' vs. ' + conds[1].split('/')[1] + '_' + self.chan[c] + '.pdf'),format='pdf')          # ax[2,0].set_suptitle('Condition difference')
        
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
            plt.savefig(fname=os.path.join(self.plotDir,conds[0].split('/')[1] + ' vs. ' + conds[1].split('/')[1] + '.pdf'),format='pdf')           # ax[2,0].set_suptitle('Condition difference')

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
            for argument in ['output']:
                value = kwargs.pop(argument,'power')
                setattr(self, argument, value)

        # first create h5-filename for time-frequency data
        tf_filename = self.epochFilename.split('/')[-1][:-4] + '-tfr.h5'

        # Number of cycles dependent on frequency-band
        n_cycles = freqs/2.
        # Run tf-decomposition
        if method == 'morlet':
            self.tf = tfr_morlet(self.epochs, freqs, n_cycles = n_cycles,decim=self.decim, use_fft=self.fft, return_itc=self.itc, average = self.average,output=self.output)
        elif method == 'multitaper':
            self.bandwidth = self.bandwidth if self.bandwidth > 2 else 4
            self.tf = tfr_multitaper(self.epochs, freqs, time_bandwith=self.bandwidth, n_cycles = n_cycles,decim=self.decim, use_fft=self.fft, return_itc=self.itc, average = self.average)

        # baseline if necesarry
        if self.baseline_lim:
            self.tf = self.tf.apply_baseline(mode=self.baseline_method, baseline=self.baseline_lim)
            self.tf.info['baseline']=[self.baseline_lim,self.baseline_method]
        # Crop if necessary
        if self.lims:
            self.tf.crop(tmin=self.lims[0],tmax=self.lims[1])

        # Save tfr-epoch file
        self.tf.save(self.epochFilename.split('_epo')[0] +'_epo-tfr.h5'  , overwrite=True)

        # Since exact event ids are not saved in tfr-epoch file, create separate pd Series with event numbers per tfr-epoch
        self.events = pd.Series(self.epochs.events[:,2])
        self.events.to_csv('/'+'/'.join(self.epochFilename.split('/')[1:-1])+'/'+tf_filename[:-3] + '.csv')

    def concatenateEpochs(self):
        epochFiles = glob.glob(os.path.join(self.baseDir, 'Proc', self.task, self.subject, self.subject + '_' + '[!merged]*_epo.fixf'))                  
        eps = []
        for f in epochFiles:
            eps.append(mne.read_epochs(f, preload=True))

        mergedEps = mne.concatenate_epochs(eps)
        filepath = '/'.join((f.split('/')[:-1]   ))
        fileParts = f.split('/')[-1].split('_')         
        fileParts[1] = 'merged'
        newFilename = '_'.join((fileParts))
        mergedFile = os.path.join(filepath,newFilename)
        mergedEps.save(mergedFile,overwrite=True)

    def jITPC(self,method,freqs,**kwargs):
        # This method calculates single trial phase coherence, according to the jackknife method proposed by Richter et 
        # al. (2015). Effectively, this single trial estimate is based on the difference between (1) ITPC calculated over all n-trials, 
        # weighted by n and (2) ITPC calculated over all-but-one trials weighted by n-1.
        filename = '/' + '/'.join(self.epochFilename.split('/')[1:-1]) + '/' + self.epochFilename.split('/')[-1][:-7] + 'jITPC'
        filename = '/Users/stijnnuiten/surfdrive/Data/perception/loc/26/26_0_loc_jITPC.npy'

        if kwargs.items():
            for argument in ['baseline_lim','baseline_method','lims','fft','itc','average']:
                value = kwargs.pop(argument, False)
                setattr(self, argument, value)
            for argument in ['decim','bandwidth']:
                value = kwargs.pop(argument, 1)
                setattr(self, argument, value)              

        # select relevant epochs (stimulus presentation)
        stimepoch = self.epochs['stim']

        # Calculate ITC for all epochs
        _, itc = tfr_morlet(stimepoch, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, decim=self.decim, n_jobs=1)        


        jITPC = []
        # Now loop over epochs, calculate all-but-one-trial and calculate jITPC 
        for ep in range(Fcomplex.data.shape[0]):
            stimepoch = self.epochs['stim']
            ep_select = stimepoch.copy().drop(ep)

            _,itc_ep = tfr_morlet(ep_select, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, decim=50, n_jobs=1)        
            jITPC.append(itc.data*len(self.epochs['stim'])-itc_ep.data*len(ep_select))

        np.save(filename,np.array(jITPC))

    def extractTFRevents(self,event_ids,ID,average=True):
        subject,index,task  = ID.split('_')
        tf_filename = self.baseDir + task + '/' + ID.split('_')[0] + '/' + ID + '_epo-tfr.h5'
        tfEvent_filename = tf_filename[:-3] + '.csv'
        tf = mne.time_frequency.read_tfrs(tf_filename)[0]
        events=pd.read_csv(tfEvent_filename,usecols=[1],header=None)

        chans, freqs, times = tf.ch_names, tf.freqs, tf.times
        tfr={}
        for ev in event_ids.keys():
            tfr[ev] = tf.data[events[1].isin(event_ids[ev]),:,:,:]
            if average:
                tfr[ev] = np.mean(tfr[ev],axis=0)       
        return tfr, chans, freqs, times

    def groupTF(self,task,idx,event_ids,subs,chanSel,normalize=True, bl=[-0.2,0]):
        # This method loads in subject TF-data, extracts the relevant epoch-data,
        # calculates condition differences (t-test) and plots results.

        # Load in data (and average all trials belonging to one condition)
        tfrAll = []
        for s in subs:
            ID = s + '_' + str(idx) + '_' + task 
            [dat,chans,freqs,times] = self.extractTFRevents(event_ids,ID,average=True)
            tfrAll.append(dat)

        # Select channels and extract relevant data per condition
        picks = [chans.index(c) for c in self.chanSel[chanSel] ]
        cond1 = np.array([tfrAll[s][event_ids.keys()[0]][picks,:,:].mean(axis=0) for s in range(len(subs))])
        cond2 = np.array([tfrAll[s][event_ids.keys()[1]][picks,:,:].mean(axis=0) for s in range(len(subs))])
        
        # Normalize
        if normalize: # for now only dB
            blTimes = np.logical_and(times>bl[0],times<bl[1])
            cond1 = np.array([10 * np.log10(cond1[s,f,:]/cond1[s,f,blTimes].mean()) for s in range(len(subs)) for f in range(len(freqs)) ]).reshape(len(subs),len(freqs),len(times))
            cond2 = np.array([10 * np.log10(cond2[s,f,:]/cond2[s,f,blTimes].mean()) for s in range(len(subs)) for f in range(len(freqs)) ]).reshape(len(subs),len(freqs),len(times))

        # Perform cluster-corrected t-test
        condDiff = cond2-cond1
        diffMean = condDiff.mean(axis=0)
        t_thresh = cluster_ttest(cond2,cond1,1000, 0.05)
        x = np.linspace(0,t_thresh.shape[1], t_thresh.shape[1]*100)
        y = np.linspace(0,t_thresh.shape[0], t_thresh.shape[0]*100)

        tzero = (np.abs(0-times)).argmin()

        # plot
        plt.imshow(diffMean,cmap = 'RdBu_r',origin = 'lower',vmin=-abs(diffMean).max(), vmax = abs(diffMean).max())
        colorbar()
        plt.contour(t_thresh,[0.5],colors='black',linestyles='solid',linewidths=[2],levels=1,origin='lower',extent=[0-0.5, x[:-1].max()-0.5,0-0.5, y[:-1].max()-0.5])

        yticks(range(0,len(freqs),5), np.around(freqs[::5],decimals=1),fontsize=12,fontweight='light')
        ylabel('Frequency (Hz)',fontsize=14,fontweight='bold')
        xticks(range(tzero,len(times),5), np.around(times[tzero::5],decimals=1),fontsize=12,fontweight='bold')
        xlabel('Time (s)',fontsize=14,fontweight='bold')
        title(event_ids.keys()[1] + ' - ' + event_ids.keys()[0],fontsize=16,fontweight='bold')

        plt.savefig(fname=plotDir + event_ids.keys()[1] + ' vs ' + event_ids.keys()[0] + ' group TF.pdf', format='pdf')

