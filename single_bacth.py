# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 21:07:44 2020

@author: wangm
"""

#先更改工作路径 记得使用正斜杠/
import os
os.chdir('D:/mne/1MNE_day2_preprocessing/1_single_preprocessing/')
import mne
import numpy as np
import matplotlib.pyplot as plt

#导入一例原始数据
raw =  mne.io.read_raw_brainvision('1.vhdr',preload=True)
raw.info
print(raw)
dir(raw)
sampling_rate = raw.info['sfreq']
#提取采样点信息
n_time_samps = raw.n_times
#提取时长信息 数组 换算了单位
time_secs = raw.times
ch_names = raw.ch_names
n_ch = len(ch_names)
raw.plot()
raw.plot(n_channels= 64, duration=5, scalings = 30e-6)


#通道定位
#查看数据通道信息
# print(raw.ch_names)
# montage = mne.channels.read_custom_montage('standard-10-5-cap385.elp')
# raw.set_montage(montage)

mapping = {'FP1':'Fp1', 'FPZ':'Fpz','FP2':'Fp2', 'FZ':'Fz', 
           'FCZ':'FCz', 'CZ':'Cz', 'CPZ':'CPz', 'PZ':'Pz', 
           'PO5':'PO5', 'POZ':'POz', 'PO6':'PO6', 'OZ':'Oz', 
           'HEO':'HEOG', 'VEO':'VEOG'}
raw_rename_ch = raw.copy().rename_channels(mapping)
montage = mne.channels.read_custom_montage('standard-10-5-cap385.elp')
raw_rename_ch.set_montage(montage)

#绘制地形图
raw_rename_ch.plot_sensors()
raw_rename_ch.plot_sensors(ch_type = 'eeg', kind = '3d')
raw_rename_ch.plot_sensors(ch_type = 'eeg', show_names = True, sphere = 0.0775)
plt.show()

#绘制频谱图
raw_rename_ch.plot_psd(fmin = 1, fmax = 70,spatial_colors = True)
#所有通道平均的频谱图
raw_rename_ch.plot_psd(fmin = 1, fmax = 70,average = True)

raw_select_ch = raw_rename_ch.copy()
#将通道类别改为眼电
raw_select_ch.set_channel_types({'HEOG':'eog','VEOG':'eog'})
raw_select_ch.info
#去除眼电电极
raw_select_ch.pick(['eeg'])
raw_select_ch.info

#滤波
raw_band = raw_select_ch.copy().filter(0.1,30,picks = 'eeg')
raw_band.plot_psd(fmin = 1, fmax = 70,spatial_colors = True)
raw_band.plot_psd(fmin = 1, fmax = 70,average = True)
#raw_band.plot()
raw_band.info
raw_band_notch = raw_band.notch_filter(50, notch_widths = 4)
raw_band_notch.plot_psd(fmin = 1, fmax = 70,spatial_colors = True)
raw_band_notch.plot_psd(fmin = 1, fmax = 70,average = True)
#raw_band_notch.plot()

#降采样
raw_resampled = raw_band_notch.copy().resample(sfreq = 500)
raw_resampled.info

#对marker进行处理
print(raw_resampled.annotations)
event_unic = set(raw_resampled.annotations.description)
print(event_unic)
help(mne.events_from_annotations)

events_from_anno, event_dict =\
    mne.events_from_annotations(raw_resampled)
print(event_dict)
print(events_from_anno)

custom_mapping = {'Stimulus/10': 10, 'Stimulus/11': 11}
print(event_dict)
print(events_from_anno)

(events_from_anno,event_dict) =\
    mne.events_from_annotations(raw_resampled,event_id = custom_mapping)
print(event_dict)
print(events_from_anno)

#分段
my_epochs = mne.Epochs\
    (raw_resampled, events_from_anno, tmin = -0.2, tmax = 0.8,baseline = (-0.2,0))
my_epochs.apply_baseline() #基线校正
my_epochs.info

help(mne.io.brainvision)
#保存刚分段好的数据
import pickle
output = open('my_epochs.pkl','wb')
pickle.dump(my_epochs, output)
output.close()

del my_epochs
pkl_file = open('my_epochs.pkl','rb')
my_epochs = pickle.load(pkl_file)
pkl_file.close()
my_epochs.info


#去除坏段
my_epochs_good = my_epochs.copy()
my_epochs_good.plot(n_channels= 62, n_epochs= 5, scalings = 60e-6)
my_epochs_good.drop_bad
my_epochs_good.info
print(len(my_epochs_good.events))

#插值坏电极
good_ch = my_epochs_good.load_data().copy().interpolate_bads(reset_bads=True)
good_ch.info
good_ch.plot(n_channels= 62, n_epochs= 5, scalings = 60e-6)


output = open('good_ch.pkl','wb')
pickle.dump(good_ch, output)
output.close()

pkl_file = open('good_ch.pkl','rb')
good_ch = pickle.load(pkl_file)
pkl_file.close()

#ICA
from mne.preprocessing import (ICA)
ica_data = good_ch.copy()
ica = ICA(n_components= 60)
ica.fit(ica_data)
#绘制所有成分的地形图
ica.plot_components()

good_ch.load_data()
#绘制所有成分的时间序列
ica.plot_sources(ica_data, show_scrollbars=False)
#range(0,50) [0,50)
ica.plot_properties(ica_data, picks= [0,1])
#绘制所有成分的属性图
ica.plot_properties(ica_data, picks= np.arange(60))
#对比去除成分前后的变化，只支持连续未分段数据
ica.plot_overlay(ica_data, exclude=[0], picks='eeg')

output = open('ica.pkl','wb')
pickle.dump(ica, output)
output.close()

pkl_file = open('ica.pkl','rb')
my_epochs = pickle.load(pkl_file)
pkl_file.close()

#去ICA成分
ica.exclude = [0] #输入要去的成分
ica.plot_components([0]) #绘制要去的成分
ica_clean = ica_data.copy()
#ica_clean.load_data()
ica.apply(ica_clean) #应用到ica_clean中

#对比
ica_clean.plot(n_epochs = 5, scalings = 30e-6)
ica_data.plot(n_epochs = 5, scalings = 30e-6)

output = open('ica_clean.pkl','wb')
pickle.dump(ica_clean, output)
output.close()

pkl_file = open('ica_clean.pkl','rb')
good_ch = pickle.load(pkl_file)
pkl_file.close()

#卡阈值
e_data = ica_clean.copy()
reject_criteria = dict(eeg=100e-6) #阈值
stronger_reject_criteria = dict(eeg=70e-6) #更严格的阈值
e_data.drop_bad(reject=stronger_reject_criteria) #应用卡阈值方法
print(e_data.drop_log) #各试次上超出阈值的电极
e_data.plot_drop_log() #共拒绝的各试次的情况

output = open('extreme_values.pkl','wb')
pickle.dump(e_data, output)
output.close()

pkl_file = open('extreme_values.pkl','rb')
e_data = pickle.load(pkl_file)
pkl_file.close()

#重参考
refered = e_data.copy().\
    set_eeg_reference(ref_channels=['TP9', 'TP10'])
#refered = extreme_values.copy().\
#    set_eeg_reference(ref_channels='average')

output = open('refered.pkl','wb')
pickle.dump(refered, output)
output.close()

