import os,sys
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
#import keras
import numpy as np
#from keras import backend as K
#from tensorflow.python.client import device_lib
#device_lib.list_local_devices()
from optparse import OptionParser
import argparse
import pandas as pd
import h5py
import json
import matplotlib
matplotlib.use('agg')
#%matplotlib inline
import matplotlib.pyplot as plt
from keras import layers
from keras import optimizers
from keras import metrics
from keras.layers import Input, Dense, Dropout, Activation, concatenate, BatchNormalization, GRU, Add, Conv1D, Conv2D, Concatenate
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, Nadam, SGD
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l1
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, roc_auc_score
import yaml


fColors = {
'black'    : (0.000, 0.000, 0.000), # hex:000000
'blue'     : (0.122, 0.467, 0.706), # hex:1f77b4
'orange'   : (1.000, 0.498, 0.055), # hex:ff7f0e
'green'    : (0.173, 0.627, 0.173), # hex:2ca02c
'red'      : (0.839, 0.153, 0.157), # hex:d62728
'purple'   : (0.580, 0.404, 0.741), # hex:9467bd
'brown'    : (0.549, 0.337, 0.294), # hex:8c564b
'darkgrey' : (0.498, 0.498, 0.498), # hex:7f7f7f
'olive'    : (0.737, 0.741, 0.133), # hex:bcbd22
'cyan'     : (0.090, 0.745, 0.812)  # hex:17becf
}

colorlist = ['blue','orange','green','red','purple','brown','darkgrey','cyan']

checkpoint_filepath = "models/tmp_checkpoint"

with open("./pf_old.json") as jsonfile:
    payload = json.load(jsonfile)
    weight_old = payload['weight']
    features_old = payload['features']
    altfeatures_old = payload['altfeatures']
    cut_old = payload['cut']
    ss_old = payload['ss_vars']
    label_old = payload['!decayType']

with open("./pf.json") as jsonfile:
    payload = json.load(jsonfile)
    weight = payload['weight']
    features = payload['features']
    altfeatures = payload['altfeatures']
    cut = payload['cut']
    ss = payload['ss_vars']
    label = payload['!decayType']

def signedDeltaPhi(phi1, phi2):
    dPhi = phi1 - phi2
    dPhi[dPhi < -np.pi] = dPhi[dPhi < -np.pi] + (2 * np.pi)
    dPhi[dPhi > np.pi] = dPhi[dPhi > -np.pi] - (2 * np.pi)
    return dPhi

norm_settings = {
        "MET_phi":(signedDeltaPhi,"fj_phi"),
        "PuppiMET_phi":(signedDeltaPhi,"fj_phi"),
        }

target = "fj_genMass"
target_old = "fj_msd"
#target_norm = "fj_msd"
target_norm = ""
evt_feats = ["MET_covXX","MET_covXY","MET_covYY","MET_phi","MET_pt","MET_significance","PuppiMET_pt","PuppiMET_phi","fj_eta","fj_phi","fj_msd","fj_pt"]

# columns declared in file
lColumns = weight + ss
nparts = 30
lPartfeatures = []
for iVar in features:
    for i0 in range(nparts):
        lPartfeatures.append(iVar+str(i0))
nsvs = 5
lSVfeatures = []
for iVar in altfeatures:
    for i0 in range(nsvs):
        lSVfeatures.append(iVar+str(i0))
lColumns = lColumns + lPartfeatures + lSVfeatures + [label]

# columns declared in file
lColumns_old = weight_old + ss_old
nparts_old = 30
lPartfeatures_old = []
for iVar in features_old:
    for i0 in range(nparts_old):
        lPartfeatures_old.append(iVar+str(i0))
nsvs_old = 5
lSVfeatures_old = []
for iVar in altfeatures_old:
    for i0 in range(nsvs_old):
        lSVfeatures_old.append(iVar+str(i0))
lColumns_old = lColumns_old + lPartfeatures_old + lSVfeatures_old + [label_old]

features_to_plot = weight_old + ss_old

bin_dict = {
        "fj_pt":np.arange(300.,725.,25.),
        "fj_msd":np.arange(0.,270.,20.),
        "MET_pt":np.arange(0.,270.,20.),
        }


def load(iFile,columns=lColumns,target_name=target,test_train_split=0.2,doscale=True,iNparts=30,iNSVs=5,fillGenM=None,maxevts=0):
    h5File = h5py.File(iFile)
    print(h5File)
    treeArray = h5File['deepDoubleTau'][()]
    print(treeArray)
    print(treeArray.shape)

    features_labels_df = pd.DataFrame(treeArray,columns=columns)
    #if maxevts>0: features_labels_df = features_labels_df.head(maxevts)

    if fillGenM is not None:
        features_labels_df.insert(len(columns),target_name,fillGenM*np.ones(len(features_labels_df.index)))

    for var in norm_settings:
        norm_op,norm_var = norm_settings[var]
        features_labels_df[var] = norm_op(features_labels_df[var],features_labels_df[norm_var])

    cutlist = cut.split(' && ')
    cut_var = [c.split('>')[0] for c in cutlist]
    cut_val = [c.split('>')[-1] for c in cutlist]
    for ic in range(len(cut_var)):
        features_labels_df = features_labels_df[features_labels_df[cut_var[ic]]>float(cut_val[ic])]

    idconv = {211.:1, 13.:2,  22.:3,  11.:4, 130.:5, 1.:6, 2.:7, 3.:8, 4.:9,
            5.:10, -211.:1, -13.:2,
            -11.:4, -1.:-6, -2.:7, -3.:8, -4.:9, -5.:10, 0.:0}
    nIDs = 33
    for i0 in range(nparts):
        features_labels_df['PF_id'+str(i0)] = features_labels_df['PF_id'+str(i0)].map(idconv)
    selPartfeatures = []
    for i0 in range(iNparts):
        for iVar in features:
            selPartfeatures.append(iVar+str(i0))
    selSVfeatures = []
    for i0 in range(iNSVs):
        for iVar in altfeatures:
            selSVfeatures.append(iVar+str(i0))
    selEvtfeatures = evt_feats

    if target_norm!="" and doscale:
        features_labels_df[target_name] = features_labels_df[target_name]/features_labels_df[target_norm]

    mask = np.ones(len(features_labels_df.index)).astype(bool)
    for p in selPartfeatures:
        mask = mask & np.isfinite(features_labels_df[p])
        if (np.isfinite(features_labels_df[p]).sum()<len(features_labels_df[p])): print(p,"found nan!!")
    for p in selSVfeatures:
        mask = mask & np.isfinite(features_labels_df[p])
        if (np.isfinite(features_labels_df[p]).sum()<len(features_labels_df[p])): print(p,"found nan!!")
    for p in selEvtfeatures:
        mask = mask & np.isfinite(features_labels_df[p])
        if (np.isfinite(features_labels_df[p]).sum()<len(features_labels_df[p])): print(p,"found nan!!")

    features_labels_df = features_labels_df[mask]

    features_df        = features_labels_df[selPartfeatures].values
    features_sv_df        = features_labels_df[selSVfeatures].values
    features_evt_df        = features_labels_df[selEvtfeatures].values
    #labels          = features_labels_df[label]
    labels          = features_labels_df[target_name]
    #features_val       = features_df.values
    feat_val           = features_labels_df[features_to_plot].values

    print(features_df.shape)
    features_df = features_df.reshape(-1,iNparts,len(features))
    #for inp in range(iNparts):
    #    for inf in range(len(features)):
    #        test_dat = features_df[0,inp,inf]
    #        print('PF',inp,features[inf],test_dat)
    print(features_df.shape)
    print(features_sv_df.shape)
    features_sv_df = features_sv_df.reshape(-1,iNSVs,len(altfeatures))
    #for inp in range(iNSVs):
    #    for inf in range(len(altfeatures)):
    #        test_dat = features_sv_df[0,inp,inf]
    #        print('SV',inp,altfeatures[inf],test_dat)
    print(features_sv_df.shape)
    features_evt_df = features_evt_df.reshape(-1,len(evt_feats))
    #for inf in range(len(evt_feats)):
    #    test_dat = features_evt_df[0,inf]
    #    print('Evt',inf,evt_feats[inf],test_dat)
    print(features_evt_df.shape)
    features_val = features_df
    features_sv_val = features_sv_df
    features_evt_val = features_evt_df
    labels_val = labels
    feat_val = feat_val

    #print(features_val)
    # split into random test and train subsets 
    X_train_val, X_test, Xalt_train_val, Xalt_test, Xevt_train_val, Xevt_test, y_train_val, y_test, feat_train, feat_test = train_test_split(features_val, features_sv_val, features_evt_val, labels_val, feat_val, test_size=test_train_split, random_state=42)
    #scaler = preprocessing.StandardScaler().fit(X_train_val)
    #X_train_val = scaler.transform(X_train_val)
    #X_test      = scaler.transform(X_test)
    return X_train_val[:maxevts-1], X_test, Xalt_train_val[:maxevts-1], Xalt_test, Xevt_train_val[:maxevts-1], Xevt_test, y_train_val[:maxevts-1], y_test, feat_train[:maxevts-1], feat_test


def conditional_loss_function(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred)*(1-y_true[:,0])

def model(Inputs,Inputs_alt,Inputs_evt,X_train,Xalt_train,Xevt_train,Y_train,NPARTS=30,NSV=5):
    CLR=0.001
    print(Inputs)
    print(Inputs_alt)
    print(Inputs_evt)

    norm = BatchNormalization(momentum=0.6, name='in_bnorm')(Inputs)
    conv = Conv1D(50,1,activation='relu',name='conv1')(norm)
    conv = Conv1D(50,1,activation='relu',name='conv2')(conv)

    norm_alt = BatchNormalization(momentum=0.6, name='in_bnorm_alt')(Inputs_alt)
    conv_alt = Conv1D(50,1,activation='relu',name='conv1_alt')(norm_alt)
    conv_alt = Conv1D(50,1,activation='relu',name='conv2_alt')(conv_alt)


    gru = GRU(100,activation='tanh',recurrent_activation='sigmoid',name='gru_base', reset_after=True)(conv)
    dense   = Dense(100, activation='relu')(gru)
    norm    = BatchNormalization(momentum=0.6, name='dense4_bnorm')(dense)

    gru_alt = GRU(100,activation='tanh',recurrent_activation='sigmoid',name='gru_base_alt', reset_after=True)(conv_alt)
    dense_alt   = Dense(100, activation='relu')(gru_alt)
    norm_alt    = BatchNormalization(momentum=0.6, name='dense4_bnorm_alt')(dense_alt)

    norm_evt    = BatchNormalization(momentum=0.6, name='dense4_bnorm_evt')(Inputs_evt)

    added       = Concatenate(axis=1)([norm, norm_alt, norm_evt])

    dense   = Dense(100, activation='relu')(added)
    norm    = BatchNormalization(momentum=0.6, name='dense5_bnorm')(dense)

    #dense   = Dense(200, activation='relu')(norm)
    #norm    = BatchNormalization(momentum=0.6, name='dense6_bnorm')(dense)

    #dense   = Dense(100, activation='relu')(norm)
    #norm    = BatchNormalization(momentum=0.6, name='dense7_bnorm')(dense)

    dense   = Dense(50, activation='relu')(norm)
    norm    = BatchNormalization(momentum=0.6, name='dense8_bnorm')(dense)

    dense   = Dense(10, activation='relu')(dense)
    out     = Dense(1, activation='linear', name='massreg')(dense)
    
    regression = Model(inputs=[Inputs,Inputs_alt,Inputs_evt], outputs=[out])

    #lossfunction = 'mean_squared_error'
    #lossfunction = 'mean_absolute_error'
    lossfunction = 'mean_absolute_percentage_error'
    #quantiles = [0.5]
    #regression.compile(loss=[lambda y,f: tilted_loss(quantile,y,f) for quantile in quantiles], optimizer=Adam(CLR), metrics=['mean_squared_error','mean_absolute_error','mean_absolute_percentage_error'])
    regression.compile(loss=lossfunction, optimizer=Adam(CLR), metrics=['mean_squared_error','mean_absolute_error','mean_absolute_percentage_error'])
    models={'regression' : regression}

    return models

def train(models,X_train,Xalt_train,Xevt_train,Y_train,feat_train,NEPOCHS=20,Obatch_size=1000):
    history = {}
    #model_checkpoint_callback = ModelCheckpoint(
    #    filepath=checkpoint_filepath,
    #    save_weights_only=True,
    #    monitor='val_loss',
    #    mode='min',
    #    save_best_only=True)

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.1,
        patience=10,
        verbose=0,
        mode="min",
        restore_best_weights=True)

    history["regression"] = models['regression'].fit([X_train,Xalt_train,Xevt_train],
            Y_train,epochs=NEPOCHS,verbose=1,batch_size=Obatch_size,
            #callbacks=[model_checkpoint_callback],
            callbacks=[early_stopping_callback],
            validation_split=0.2)
     
    #model.load_weights(checkpoint_filepath)

    return history

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--tag',        dest='tag',        default="",           help="tag")
    parser.add_argument('--indir',      dest='indir',      default="files",      help="indir")
    parser.add_argument('--label',      dest='label',      default="",           help="label")
    parser.add_argument('--nepoch',     dest='nepoch',     default=20,           help="nepoch",     type=int)
    parser.add_argument('--batchsize',  dest='batchsize',  default=1000,         help="batchsize",  type=int)
    parser.add_argument('--load',       dest='load',       action='store_true',  help='load')
    parser.add_argument('--loadmax',    dest='loadmax',    default=0,            help='loadmax',    type=int)
    parser.add_argument('--cmssw',      dest='cmssw',      action='store_true',  help='cmssw')
    parser.add_argument('--test',       dest='test',       action='store_true',  help='test')
    parser.add_argument('--dodist',     dest='dodist',     action='store_true',  help='dodist')
    parser.add_argument('--out',        dest='out',        default="",        help='out')
    args = parser.parse_args()

    dtype = args.tag
    full_label = args.label
    if full_label!="":
        full_label = "_"+full_label
    full_out = args.out
    if full_out!="":
        full_out = "_"+full_out

    X_train,X_test,Xalt_train,Xalt_test,Xevt_train,Xevt_test,Y_train,Y_test,feat_train,feat_test = load('%s/FlatTauTau_user_%s.z'%(args.indir,dtype),maxevts=(0 if not args.cmssw else 100))
    if args.load:
        X_train_htt,X_test_htt,Xalt_train_htt,Xalt_test_htt,Xevt_train_htt,Xevt_test_htt,Y_train_htt,Y_test_htt,feat_train_htt,feat_test_htt = load('%s/GluGluHToTauTau_user_%s.z'%(args.indir,dtype),columns=lColumns_old,fillGenM=125.,maxevts=args.loadmax)
        X_train_ztt,X_test_ztt,Xalt_train_ztt,Xalt_test_ztt,Xevt_train_ztt,Xevt_test_ztt,Y_train_ztt,Y_test_ztt,feat_train_ztt,feat_test_ztt = load('%s/DYJetsToLL_%s.z'%(args.indir,dtype),columns=lColumns_old,fillGenM=91.,maxevts=args.loadmax)

        X_train = np.concatenate([X_train, X_train_htt, X_train_ztt])
        Xalt_train = np.concatenate([Xalt_train, Xalt_train_htt, Xalt_train_ztt])
        Xevt_train = np.concatenate([Xevt_train, Xevt_train_htt, Xevt_train_ztt])
        Y_train = np.concatenate([Y_train, Y_train_htt, Y_train_ztt])
        feat_train = np.concatenate([feat_train, feat_train_htt, feat_train_ztt])
        X_test = np.concatenate([X_test, X_test_htt, X_test_ztt])
        Xalt_test = np.concatenate([Xalt_test, Xalt_test_htt, Xalt_test_ztt])
        Xevt_test = np.concatenate([Xevt_test, Xevt_test_htt, Xevt_test_ztt])
        Y_test = np.concatenate([Y_test, Y_test_htt, Y_test_ztt])
        feat_test = np.concatenate([feat_test, feat_test_htt, feat_test_ztt])

    mbins = np.arange(-1.40,1.44,0.04)

    genMass_train = Y_train
    genMass_test = Y_test
    if target_norm!="":
        genMass_train = genMass_train*feat_train[:,weight.index(target_norm)]
        genMass_test = genMass_test*feat_test[:,weight.index(target_norm)]

    inputvars=Input(shape=X_train.shape[1:], name='input')
    inputvars_alt=Input(shape=Xalt_train.shape[1:], name='altinput')
    inputvars_evt=Input(shape=Xevt_train.shape[1:], name='evtinput')
    models = model(inputvars,inputvars_alt,inputvars_evt,X_train,Xalt_train,Xevt_train,Y_train)
    for m in models:
        print(str(m), models[m].summary())
    if not args.cmssw:
        history = train(models,X_train,Xalt_train,Xevt_train,Y_train,feat_train,args.nepoch,args.batchsize)
    else:
        history = train(models,X_train[:10],Xalt_train[:10],Xevt_train[:10],Y_train[:10],feat_train[:10],1,10)
    #print(len(Y_test),' vs ',sum(Y_test))
    #test(models,X_test,Y_test,feat_test)
    #from keras.backend import manual_variable_initialization manual_variable_initialization(True)
    for m in models:

        model_json = models[m].to_json()
        with open("models/model"+full_label+"_"+dtype+"_"+str(m)+".json", "w") as json_file:
            json_file.write(model_json)

        if args.cmssw:
            models[m].load_weights("models/model"+full_label+"_"+dtype+"_"+str(m)+".h5")
            print(str(m), models[m].summary())
            #print(models[m].get_weights())

        else:

            models[m].save_weights("models/model"+full_label+"_"+dtype+"_"+str(m)+".h5")
            print(history[str(m)].history.keys())
            for k in history[str(m)].history.keys():
                if k.startswith('val'): continue
   
                plt.clf() 
                plt.plot(history[str(m)].history['%s'%k])
                plt.plot(history[str(m)].history['val_%s'%k])
                plt.title('model %s'%k)
                plt.ylabel('%s'%k)
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.savefig("plots/%s%s_%s_%s.pdf"%(dtype,full_label,str(m),k))

        models[m].save("models/fullmodel_"+dtype+"_"+str(m)+(full_label if full_out=="" else full_out)+".h5")

        print(X_test[:10],Xalt_test[:10],Xevt_test[:10])

        Y_pred_all = models[m].predict([X_test[:10],Xalt_test[:10],Xevt_test[:10]])
        print(Y_pred_all)
        Y_pred = Y_pred_all.flatten()

        print(Y_test[:10])
        print(Y_pred[:10])
        if target_norm!="": print(Y_pred[:10]*feat_test[:10,weight.index(target_norm)])
        print(genMass_test[:10])

        if args.test:
            print('-----------------')
            print('Running CMSSW check')
            with open('test_one.npy', 'rb') as f:
                pfData = np.load(f)
                svData = np.load(f)
                evtData = np.load(f)
                hadelmuNN = np.load(f)
    
                hadhad_w = np.load(f,allow_pickle=True,encoding='bytes')
                hadel_w = np.load(f,allow_pickle=True,encoding='bytes')
                hadmu_w = np.load(f,allow_pickle=True,encoding='bytes')
    
            thisModel = models[m].predict([pfData,svData,evtData])
    
            print(thisModel,'vs',hadelmuNN)
    
            these_w = models[m].get_weights()
            load_w = hadel_w if dtype=='hadel' else hadmu_w if dtype=='hadmu' else hadhad_w
    
            n_w = len(load_w)
            diff_w = 0
    
            for ar in range(n_w):
                if not np.array_equal(load_w[ar],these_w[ar]):
                    diff_w = diff_w + 1
                    print(ar,':',load_w[ar],'vs',these_w[ar])
    
            print(diff_w,'/',n_w)

        if args.dodist:
            pfData = []
            svData = []
            evtData = []
            hadelmuNN = []
            with open('test.npy', 'rb') as f:
                try:
                    pfData.append(np.load(f))
                except:
                    break
                svData.append(np.load(f))
                evtData.append(np.load(f))
                hadelmuNN.append(np.load(f))
            pfData = np.vstack(pfData)
            svData = np.vstack(svData)
            evtData = np.vstack(evtData)
            hadelmuNN = np.vstack(hadelmuNN)
            theNN = hadelmuNN[:,0 if dtype=="hadhad" else 1 if dtype=="hadel" else 2]

            Y_pred = models[m].predict([X_test,Xalt_test,Xevt_test]).flatten()
            theNN = models[m].predict([pfData,svData,evtData]).flatten()

            print(evtData.shape)
            #print({evt_feats[ic]:evtData[0][ic] for ic in range(len(evt_feats))})
    
            for ic,col in enumerate(evt_feats):
                plt.clf() 
                _,thebins,_ = plt.hist(Xevt_test[:,ic],histtype='step',density=True,fill=False)
                thebins = np.linspace(thebins[0] if np.quantile(Xevt_test[:,ic],0.05)<(thebins[0]+(thebins[-1]-thebins[0])*0.25) else np.quantile(Xevt_test[:,ic],0.05),thebins[-1] if np.quantile(Xevt_test[:,ic],0.95)>(thebins[0]+(thebins[-1]-thebins[0])*0.75) else np.quantile(Xevt_test[:,ic],0.95))
                plt.clf()
                plt.hist(Xevt_test[:,ic],bins=thebins,histtype='step',density=True,fill=False)
                plt.hist(Xevt_test_htt[:,ic],bins=thebins,histtype='step',density=True,fill=False)
                plt.hist(Xevt_test_ztt[:,ic],bins=thebins,histtype='step',density=True,fill=False)
                plt.hist(evtData[:,ic][np.abs(evtData[:,evt_feats.index('MET_phi')])<0.5],bins=thebins,histtype='step',density=True,fill=False)
                plt.ylabel('arb.')
                plt.xlabel(col)
                plt.legend(['Flat Htt', 'ggHtt', 'Ztt', 'CMSSW'], loc='best')
                plt.savefig("plots/varcomp_%s_%s.pdf"%(dtype,col))
                plt.yscale('log')
                plt.savefig("plots/varcomp_%s_%s_logy.pdf"%(dtype,col))

                plt.clf()
                plt.hist2d(Xevt_test[:,ic],Y_pred,bins=[thebins,np.arange(10.,510.,10.)],density=True)
                plt.plot(evtData[:,ic][np.abs(evtData[:,evt_feats.index('MET_phi')])<0.5],theNN[np.abs(evtData[:,evt_feats.index('MET_phi')])<0.5],'rx')
                plt.ylabel('mreg')
                plt.xlabel(col)
                plt.savefig("plots/varcomp2d_%s_%s.pdf"%(dtype,col))
                #plt.zscale('log')
                #plt.savefig("plots/varcomp2d_%s_%s_logz.pdf"%(dtype,col))
            for ic,col in enumerate(lSVfeatures):
                ipn = ic % nsvs
                ifn = int((ic - ipn)/nsvs)
                plt.clf() 
                _,thebins,_ = plt.hist(Xalt_test[:,ipn,ifn],histtype='step',density=True,fill=False)
                thebins = np.linspace(thebins[0] if np.quantile(Xalt_test[:,ipn,ifn],0.05)<(thebins[0]+(thebins[-1]-thebins[0])*0.25) else np.quantile(Xalt_test[:,ipn,ifn],0.05),thebins[-1] if np.quantile(Xalt_test[:,ipn,ifn],0.95)>(thebins[0]+(thebins[-1]-thebins[0])*0.75) else np.quantile(Xalt_test[:,ipn,ifn],0.95))
                plt.clf()
                plt.hist(Xalt_test[:,ipn,ifn],bins=thebins,histtype='step',density=True,fill=False)
                plt.hist(Xalt_test_htt[:,ipn,ifn],bins=thebins,histtype='step',density=True,fill=False)
                plt.hist(Xalt_test_ztt[:,ipn,ifn],bins=thebins,histtype='step',density=True,fill=False)
                plt.hist(svData[:,ipn,ifn][np.abs(evtData[:,evt_feats.index('MET_phi')])<0.5],bins=thebins,histtype='step',density=True,fill=False)
                plt.ylabel('arb.')
                plt.xlabel(col)
                plt.legend(['Flat Htt', 'ggHtt', 'Ztt', 'CMSSW'], loc='best')
                plt.savefig("plots/varcomp_%s_%s.pdf"%(dtype,col))
                plt.yscale('log')
                plt.savefig("plots/varcomp_%s_%s_logy.pdf"%(dtype,col))

                plt.clf()
                plt.hist2d(Xalt_test[:,ipn,ifn],Y_pred,bins=[thebins,np.arange(10.,510.,10.)],density=True)
                plt.plot(svData[:,ipn,ifn][np.abs(evtData[:,evt_feats.index('MET_phi')])<0.5],theNN[np.abs(evtData[:,evt_feats.index('MET_phi')])<0.5],'rx')
                plt.ylabel('mreg')
                plt.xlabel(col)
                plt.savefig("plots/varcomp2d_%s_%s.pdf"%(dtype,col))
                #plt.zscale('log')
                #plt.savefig("plots/varcomp2d_%s_%s_logz.pdf"%(dtype,col))
            print(X_test.shape)
            print(pfData.shape)
            for ic,col in enumerate(lPartfeatures):
                ipn = ic % nparts
                ifn = int((ic - ipn)/nparts)
                #print(col,ipn,ifn,pfData[:,ipn,ifn][np.abs(evtData[:,evt_feats.index('MET_phi')])<0.5])
                plt.clf() 
                _,thebins,_ = plt.hist(X_test[:,ipn,ifn],histtype='step',density=True,fill=False)
                thebins = np.linspace(thebins[0] if np.quantile(X_test[:,ipn,ifn],0.05)<(thebins[0]+(thebins[-1]-thebins[0])*0.25) else np.quantile(X_test[:,ipn,ifn],0.05),thebins[-1] if np.quantile(X_test[:,ipn,ifn],0.95)>(thebins[0]+(thebins[-1]-thebins[0])*0.75) else np.quantile(X_test[:,ipn,ifn],0.95))
                plt.clf()
                plt.hist(X_test[:,ipn,ifn],bins=thebins,histtype='step',density=True,fill=False)
                plt.hist(X_test_htt[:,ipn,ifn],bins=thebins,histtype='step',density=True,fill=False)
                plt.hist(X_test_ztt[:,ipn,ifn],bins=thebins,histtype='step',density=True,fill=False)
                plt.hist(pfData[:,ipn,ifn][np.abs(evtData[:,evt_feats.index('MET_phi')])<0.5],bins=thebins,histtype='step',density=True,fill=False)
                plt.ylabel('arb.')
                plt.xlabel(col)
                plt.legend(['Flat Htt', 'ggHtt', 'Ztt', 'CMSSW'], loc='best')
                plt.savefig("plots/varcomp_%s_%s.pdf"%(dtype,col))
                plt.yscale('log')
                plt.savefig("plots/varcomp_%s_%s_logy.pdf"%(dtype,col))

                plt.clf()
                plt.hist2d(X_test[:,ipn,ifn],Y_pred,bins=[thebins,np.arange(10.,510.,10.)],density=True)
                plt.plot(pfData[:,ipn,ifn][np.abs(evtData[:,evt_feats.index('MET_phi')])<0.5],theNN[np.abs(evtData[:,evt_feats.index('MET_phi')])<0.5],'rx')
                plt.ylabel('mreg')
                plt.xlabel(col)
                plt.savefig("plots/varcomp2d_%s_%s.pdf"%(dtype,col))
                #plt.zscale('log')
                #plt.savefig("plots/varcomp2d_%s_%s_logz.pdf"%(dtype,col))

            plt.clf()
            thebins = np.arange(10.,510.,10.)
            plt.hist(Y_pred,bins=thebins,histtype='step',density=True,fill=False)
            plt.hist(theNN[np.abs(evtData[:,evt_feats.index('MET_phi')])<0.5],bins=thebins,histtype='step',density=True,fill=False)
            plt.ylabel('arb.')
            plt.xlabel('NN mass')
            plt.legend(['Flat Htt', 'CMSSW'], loc='best')
            plt.savefig("plots/varcomp_%s_nnmass.pdf"%(dtype))
            plt.yscale('log')
            plt.savefig("plots/varcomp_%s_nnmass_logy.pdf"%(dtype))
