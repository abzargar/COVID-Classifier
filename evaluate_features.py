# =============================================================================
# evaluate extracted features on mat files
# =============================================================================
import scipy.io as sio
import os
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc , classification_report
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# =============================================================================
# source dir
# =============================================================================
source_dir='./'
# =============================================================================
# load mat files
# =============================================================================
covid_features=sio.loadmat(os.path.join(source_dir,'covid.mat')) 
covid_features=covid_features['covid'] 

normal_features=sio.loadmat(os.path.join(source_dir,'normal.mat')) 
normal_features=normal_features['normal']  

pneumonia_features=sio.loadmat(os.path.join(source_dir,'pneumonia.mat')) 
pneumonia_features=pneumonia_features['pneumonia']    

scores=np.concatenate((covid_features[:,:-1],normal_features[:,:-1],pneumonia_features[:,:-1]), axis=0)
targets=np.concatenate((covid_features[:,-1],normal_features[:,-1],pneumonia_features[:,-1]), axis=0)
# =============================================================================
# Normalization
# =============================================================================
min_max_scaler=MinMaxScaler()
scores = min_max_scaler.fit_transform(scores) 
# =============================================================================
# correlation map and histogram 
# =============================================================================
df = pd.DataFrame(scores)             
corrMatrix = df.corr().abs().fillna(0)
fig = plt.figure()
sn.heatmap(corrMatrix,xticklabels=50,yticklabels=50,cmap='jet')
plt.savefig('corr_map',dpi=300,format='eps')
# 
fig = plt.figure()
(h,x)=np.histogram(corrMatrix.to_numpy().reshape(1,252*252), bins=8) 
plt.bar(x[1:],h/(252*252),width=0.1)            
plt.grid(linewidth=.3)  
plt.xlabel('Correlation coefficient value')
plt.ylabel('Frequency percentage (%)')
plt.savefig('corr_hist.jpg',dpi=300)
# =============================================================================
# auc value graphs where positive label is COVID-19             
# =============================================================================
pos_label=0   
roc_list=[]  
for idx in range(scores.shape[1]):               
    fpr, tpr, thresholds = roc_curve(targets, scores[:,idx], pos_label=pos_label)
    auc_value=auc(fpr, tpr)
    if auc_value<0.5:
        auc_value=1-auc_value
    roc_list.append(auc_value)# calculate auc value

print(np.argmax(roc_list))
print(np.max(roc_list))
fig = plt.figure()
plt.bar([1,2,3,4,5],[np.mean(roc_list[:14]),np.mean(roc_list[14:28])
        ,np.mean(roc_list[28:140]),np.mean(roc_list[140:196]),np.mean(roc_list[196:252])]
        ,width=0.5)

plt.xticks([1,2,3,4,5], ('Texture', 'FFT', 'Wavelet', 'GLDM', 'GLCM'))
plt.grid(linewidth=.3)
plt.ylim([0, 1])
plt.title('pos_label=COVID-19')
plt.ylabel('Average AUC value')
plt.savefig('avg_bar_pos_COVID.jpg',dpi=300)

fig = plt.figure()
plt.plot([i for i in range(1,scores.shape[1]+1)],np.sort(roc_list)[::-1],'x')
plt.grid(linewidth=.3)
plt.xlim([0.0, scores.shape[1]+1])
plt.ylim([0.5, 1])
plt.xlabel('Features')
plt.ylabel('Sorted AUC values')
plt.title('pos_label=COVID-19')
plt.savefig('auc_pos_COVID.jpg',dpi=300)

# =============================================================================
# auc value graphs where positive label is normal
# =============================================================================
pos_label=1   
roc_list=[]  
for idx in range(scores.shape[1]):               
    fpr, tpr, thresholds = roc_curve(targets, scores[:,idx], pos_label=pos_label)
    auc_value=auc(fpr, tpr)
    if auc_value<0.5:
        auc_value=1-auc_value
    roc_list.append(auc_value)# calculate auc value

fig = plt.figure()
plt.bar([1,2,3,4,5],[np.mean(roc_list[:14]),np.mean(roc_list[14:28])
        ,np.mean(roc_list[28:140]),np.mean(roc_list[140:196]),np.mean(roc_list[196:252])]
        ,width=0.5)

plt.xticks([1,2,3,4,5], ('Texture', 'FFT', 'Wavelet', 'GLDM', 'GLCM'))
plt.grid(linewidth=.3)
plt.ylim([0, 1])
plt.title('pos_label=Normal')
plt.ylabel('Average AUC value')
plt.savefig('avg_bar_pos_Normal.jpg',dpi=300)

print(np.argmax(roc_list))
print(np.max(roc_list))
fig = plt.figure()
plt.plot([i for i in range(1,scores.shape[1]+1)],np.sort(roc_list)[::-1],'x')
plt.grid(linewidth=.3)
plt.xlim([0.0, scores.shape[1]+1])
plt.ylim([0.5, 1])
plt.xlabel('Features')
plt.ylabel('Sorted AUC values')
plt.title('pos_label=Normal')
plt.savefig('auc_pos_Normal.jpg',dpi=300)

# =============================================================================
# auc value graphs where positive label is pneumonia
# =============================================================================
pos_label=2   
roc_list=[]  
for idx in range(scores.shape[1]):               
    fpr, tpr, thresholds = roc_curve(targets, scores[:,idx], pos_label=pos_label)
    auc_value=auc(fpr, tpr)
    if auc_value<0.5:
        auc_value=1-auc_value
    roc_list.append(auc_value)# calculate auc value

print(np.argmax(roc_list))
print(np.max(roc_list))

fig = plt.figure()
plt.bar([1,2,3,4,5],[np.mean(roc_list[:14]),np.mean(roc_list[14:28])
        ,np.mean(roc_list[28:140]),np.mean(roc_list[140:196]),np.mean(roc_list[196:252])]
        ,width=0.5)

plt.xticks([1,2,3,4,5], ('Texture', 'FFT', 'Wavelet', 'GLDM', 'GLCM'))
plt.grid(linewidth=.3)
plt.ylim([0, 1])
plt.title('pos_label=Pneumonia')
plt.ylabel('Average AUC value')
plt.savefig('avg_bar_pos_Pneumonia.jpg',dpi=300)

fig = plt.figure()
plt.plot([i for i in range(1,scores.shape[1]+1)],np.sort(roc_list)[::-1],'x')
plt.grid(linewidth=.3)
plt.xlim([0.0, scores.shape[1]+1])
plt.ylim([0.5, 1])
plt.xlabel('Features')
plt.ylabel('Sorted AUC values')
plt.title('pos_label=Pneumonia')
plt.savefig('auc_pos_Pneumonia.jpg',dpi=300)
