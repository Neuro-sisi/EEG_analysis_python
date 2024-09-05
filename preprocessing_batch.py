# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 23:46:47 2020

@author: wangm
"""

#先更改工作路径 记得使用正斜杠/
import os
os.chdir("D:/mne/1MNE_day2_preprocessing/2_preprocessing_batch/")
import mne
import numpy as np
import matplotlib.pyplot as plt
import pickle

root = 'D:/mne/1MNE_day2_preprocessing/2_preprocessing_batch/'
path_list = ['1_raw_data/','2_epoch_data/','3_rm_epochs_g_chs/',
             '4_ica_cleaned/','5_extreme_values_refered/']

for i in range(1,3):
    filename = os.path.join(root,path_list[0]) + str(i) + '.vhdr'
    print(filename)

    raw =  mne.io.read_raw_brainvision(filename,preload=True)
    mapping = {'FP1':'Fp1', 'FPZ':'Fpz','FP2':'Fp2', 'FZ':'Fz', 
           'FCZ':'FCz', 'CZ':'Cz', 'CPZ':'CPz', 'PZ':'Pz', 
           'PO5':'PO5', 'POZ':'POz', 'PO6':'PO6', 'OZ':'Oz', 
           'HEO':'HEOG', 'VEO':'VEOG'}
    raw_rename_ch = raw.copy().rename_channels(mapping)
    montage = mne.channels.read_custom_montage('standard-10-5-cap385.elp')
    raw_rename_ch.set_montage(montage)
    raw_select_ch = raw_rename_ch.copy()
    raw_select_ch.set_channel_types({'HEOG':'eog','VEOG':'eog'})
    raw_select_ch.pick(['eeg'])


    #滤波
    raw_band = raw_select_ch.copy().filter(0.1,30,picks = 'eeg')
    raw_band_notch = raw_band.notch_filter(50, notch_widths = 4)
    raw_resampled = raw_band_notch.copy().resample(sfreq = 500)
    #分段
    custom_mapping = {'Stimulus/10': 10, 'Stimulus/11': 11}
    (events_from_anno,event_dict) = mne.events_from_annotations(raw_resampled,event_id = custom_mapping)
    my_epochs = mne.Epochs(raw_resampled, events_from_anno, tmin = -0.2, tmax = 0.8,baseline = (-0.2,0))
    savename = os.path.join(root,path_list[1]) + str(i) + '.pkl'

    output = open(savename,'wb')
    pickle.dump(my_epochs, output)
    output.close()


i = 1
filename = os.path.join(root,path_list[1]) + str(i) + '.pkl'
pkl_file = open(filename,'rb')
my_epochs = pickle.load(pkl_file)
pkl_file.close()
    

#去除坏段
my_epochs_good = my_epochs.copy()
my_epochs_good.plot(n_epochs = 5)
my_epochs_good.drop_bad
my_epochs_good.info

#插值坏电极
good_ch = my_epochs_good.load_data().copy().interpolate_bads(reset_bads=False)
savename = os.path.join(root,path_list[2]) + str(i) + '.pkl'
output = open(savename,'wb')
pickle.dump(good_ch, output)
output.close()


from mne.preprocessing import (ICA)
i = 1
filename = os.path.join(root,path_list[2]) + str(i) + '.pkl'
pkl_file = open(filename,'rb')
good_ch = pickle.load(pkl_file)
pkl_file.close()

ica_data = good_ch.copy()
ica = ICA(n_components= 50)
ica.fit(ica_data)
ica.plot_components()

good_ch.load_data()
ica.plot_sources(good_ch, show_scrollbars=False)
#range(0,50) [0,50)
ica.plot_properties(good_ch, picks= [0,1])
ica.plot_properties(good_ch, picks= np.array(range(0,50)))
ica.plot_overlay(raw_resampled, exclude=[0], picks='eeg')

ica.exclude = [0]
ica_clean = good_ch.copy()
ica_clean.load_data()
ica.apply(ica_clean)

ica_clean.plot(n_epochs = 5, scalings = 30e-6)
good_ch.plot(n_epochs = 5, scalings = 30e-6)


savename = os.path.join(root,path_list[3]) + str(i) + '.pkl'
output = open(savename,'wb')
pickle.dump(ica_clean, output)
output.close()

for i in range(1,3):
    filename = os.path.join(root,path_list[3]) + str(i) + '.pkl'
    pkl_file = open(filename,'rb')
    ica_clean = pickle.load(pkl_file)
    pkl_file.close()

    reject_criteria = dict(eeg=100e-6)
    extreme_values = ica_clean.copy()
    stronger_reject_criteria = dict(eeg=100e-6)
    extreme_values.drop_bad(reject=stronger_reject_criteria)
    refered = extreme_values.copy().set_eeg_reference(ref_channels=['TP9', 'TP10'])
    #refered = extreme_values.copy().set_eeg_reference(ref_channels='average')

    savename = os.path.join(root,path_list[4]) + str(i) + '.pkl'
    output = open(savename,'wb')
    pickle.dump(refered, output)
    output.close()
    
    

