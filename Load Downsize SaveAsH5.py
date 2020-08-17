import numpy as np
import h5py
import pandas as pd
import os
import itertools

def downsize(df,size,seed,labels):
    jet_dict = {}
    np.random.seed(seed)
    for label in labels:
        jet_dict[label] = np.random.choice(a=df[df[label]==1].j_index, size=size,replace=False )
    mini_df = pd.DataFrame()
    for label in labels:
        mini_df = pd.concat([mini_df,df[df.j_index.isin(jet_dict[label])]],axis=0)
    return mini_df

def loadNdownsize (filePath,features,labels,size,seed):
    '''
    features: 'j_index' is necessary!
    '''
    with h5py.File(filePath+"processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_withPars_truth_0.z", 'r') as f:
        treeArray = f['t_allpar_new'][()]
    df = pd.DataFrame(treeArray, columns=features+labels)  
    mini_df = downsize(df,size,seed,labels)
    # Add constituents_index
    x = [len(list(y)) for _,y in itertools.groupby(mini_df['j_index'])]
    j_cIndex = np.array([],dtype='int8')
    for i in x:
        new_jet_index = np.arange(i)
        j_cIndex = np.append(j_cIndex, new_jet_index)
    mini_df['constituents_index'] = j_cIndex
    _features = features+['constituents_index']
    return mini_df, _features

def saveAsH5(filePath,df, size,features,labels):
    feats = features
    feats.remove('j_index')
    
    # Since list is not valid in Dataframe, create a nparray to store the label list
    label_list = np.vstack([df[label] for label in labels]).T
    
    savePath = filePath+"data_%sjets_%dlabels"%(size,len(labels))+'.h5'
    if os.path.exists(savePath):
        with h5py.File(savePath, 'r+') as f:
            for col in feats:
                f[col][()] = df[col]
            f['label'][()] = label_list
    else:
        with h5py.File(savePath, 'w') as f:
            for col in feats:
                f.create_dataset(col, data=df[col])
            f.create_dataset('label', data = label_list)

def LoadTransSave (filePath, features, labels, size, seed):
    mini_df,_features = loadNdownsize(filePath,features, labels,size=size,seed=seed)
    saveAsH5(filePath, df = mini_df, size=size, features=_features, labels=labels)

# To test:
# filePath = './data/'
# features = ['index','j1_pt','j1_eta','j1_phi','j_mass','j_multiplicity','j1_etarot','j1_phirot','j_pt','j1_etarel','j1_phirel','j_index']
# labels = ['j_g','j_q','j_w','j_z','j_t']
# size=20
# seed=42
# LoadTransSave(filePath,features, labels,size=size,seed=seed)
