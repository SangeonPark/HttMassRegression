import os,sys
#os.environ['KERAS_BACKEND'] = 'tensorflow'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, Nadam, SGD
from keras.utils import to_categorical
import matplotlib
matplotlib.use('agg')
#%matplotlib inline
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Dropout, Activation, concatenate, BatchNormalization, GRU, Add, Conv1D, Conv2D, Concatenate
from keras.models import Model 
from keras.models import model_from_json
from keras.models import load_model
from keras.regularizers import l1
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, roc_auc_score
import yaml

from train_GRU import load, target, target_old, target_norm, evt_feats, bin_dict, signedDeltaPhi, norm_settings

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

def turnon(iD,iTrainable,iOther=0):
    i0 = -1
    for l1 in iD.layers:
        i0=i0+1
        if iOther != 0 and l1 in iOther.layers:
            continue
        try:
            l1.trainable = iTrainable
        except:
            print("trainableErr",layer)

def conditional_loss_function(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred)*(1-y_true[:,0])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--tag',        dest='tag',        default="",           help="tag")
    parser.add_argument('--indir',      dest='indir',      default="files",      help="indir")
    parser.add_argument('--label',      dest='label',      default="",           help="label")
    parser.add_argument('--load',       dest='load',       action='store_true',  help='load')
    parser.add_argument('--loadspec',   dest='loadspec',   action='store_true',  help='loadspec')
    parser.add_argument('--full',       dest='full',       action='store_true',  help='full')
    args = parser.parse_args()

    dtype = args.tag
    full_label = args.label
    if full_label!="":
        full_label = "_"+full_label

    X_train,X_test,Xalt_train,Xalt_test,Xevt_train,Xevt_test,Y_train,Y_test,feat_train,feat_test = load('%s/FlatTauTau_user_%s.z'%(args.indir,dtype))
    _,X_test_htt,_,Xalt_test_htt,_,Xevt_test_htt,_,Y_test_htt,_,feat_test_htt = load('%s/GluGluHToTauTau_user_%s.z'%(args.indir,dtype),columns=lColumns_old,fillGenM=125.)
    if args.load:
        print('Loading...')
        if args.loadspec:
            if dtype=="hadhad":
                print('Getting QCD ...')
                _,X_test_qcd,_,Xalt_test_qcd,_,Xevt_test_qcd,_,Y_test_qcd,_,feat_test_qcd = load('%s/QCD_%s.z'%(args.indir,dtype),columns=lColumns_old,target_name='fj_msd',test_train_split=0.9,doscale=False)
                print('Got QCD ...')
            else:
                print('Getting WJets ...')
                _,X_test_wjets,_,Xalt_test_wjets,_,Xevt_test_wjets,_,Y_test_wjets,_,feat_test_wjets = load('%s/WJets_%s.z'%(args.indir,dtype),columns=lColumns_old,target_name='fj_msd',test_train_split=0.9,doscale=False)
                print('Got WJets ...')
                if dtype=="hadel":
                    print('Getting Zee ...')
                    _,X_test_zee,_,Xalt_test_zee,_,Xevt_test_zee,_,Y_test_zee,_,feat_test_zee = load('%s/ee-DYJetsToLL_%s.z'%(args.indir,dtype),columns=lColumns_old,target_name='fj_msd',test_train_split=0.9,doscale=False)
                    print('Got Zee ...')
                if dtype=="hadmu":
                    print('Getting Zmm ...')
                    _,X_test_zmm,_,Xalt_test_zmm,_,Xevt_test_zmm,_,Y_test_zmm,_,feat_test_zmm = load('%s/mm-DYJetsToLL_%s.z'%(args.indir,dtype),columns=lColumns_old,target_name='fj_msd',test_train_split=0.9,doscale=False)
                    print('Got Zmm ...')
            print('Getting TTbar ...')
            _,X_test_ttbar,_,Xalt_test_ttbar,_,Xevt_test_ttbar,_,Y_test_ttbar,_,feat_test_ttbar = load('%s/TTbar_%s.z'%(args.indir,dtype),columns=lColumns_old,target_name='fj_msd',test_train_split=0.9,doscale=False)
            print('Got TTbar ...')
        print('Getting Ztt ...')
        _,X_test_ztt,_,Xalt_test_ztt,_,Xevt_test_ztt,_,Y_test_ztt,_,feat_test_ztt = load('%s/DYJetsToLL_%s.z'%(args.indir,dtype),columns=lColumns_old,fillGenM=91.)
        print('Got Ztt ...')


    mbins = np.arange(-1.40,1.44,0.04)

    genMass_train = Y_train
    genMass_test = Y_test
    if target_norm!="":
        genMass_train = genMass_train*feat_train[:,weight.index(target_norm)]
        genMass_test = genMass_test*feat_test[:,weight.index(target_norm)]

    models = {}
    for m in ['regression']:
        if args.full:
            models[m] = load_model("models/fullmodel_"+dtype+"_"+str(m)+full_label+".h5")
        else:
            json_file = open('models/model'+full_label+'_'+dtype+'_'+str(m)+'.json', 'r')
            model_json = json_file.read()
            models[m] = model_from_json(model_json)
            models[m].load_weights("models/model"+full_label+"_"+dtype+"_"+str(m)+".h5")

    for m in models:

        Y_pred_all = models[m].predict([X_test,Xalt_test,Xevt_test])
        Y_pred = Y_pred_all.flatten()
        Y_pred_all_htt = models[m].predict([X_test_htt,Xalt_test_htt,Xevt_test_htt])
        Y_pred_htt = Y_pred_all_htt.flatten()
    
        if args.load:
            if args.loadspec:
                if dtype=="hadhad":
                    Y_pred_all_qcd = models[m].predict([X_test_qcd,Xalt_test_qcd,Xevt_test_qcd])
                    Y_pred_qcd = Y_pred_all_qcd.flatten()
                else:
                    Y_pred_all_wjets = models[m].predict([X_test_wjets,Xalt_test_wjets,Xevt_test_wjets])
                    Y_pred_wjets = Y_pred_all_wjets.flatten()
                    if dtype=="hadel":
                        Y_pred_all_zee = models[m].predict([X_test_zee,Xalt_test_zee,Xevt_test_zee])
                        Y_pred_zee = Y_pred_all_zee.flatten()
                    if dtype=="hadmu":
                        Y_pred_all_zmm = models[m].predict([X_test_zmm,Xalt_test_zmm,Xevt_test_zmm])
                        Y_pred_zmm = Y_pred_all_zmm.flatten()
                Y_pred_all_ttbar = models[m].predict([X_test_ttbar,Xalt_test_ttbar,Xevt_test_ttbar])
                Y_pred_ttbar = Y_pred_all_ttbar.flatten()
            Y_pred_all_ztt = models[m].predict([X_test_ztt,Xalt_test_ztt,Xevt_test_ztt])
            Y_pred_ztt = Y_pred_all_ztt.flatten()
    
    
        print(Y_test[:10])
        print(Y_pred[:10])
        if target_norm!="": print(Y_pred[:10]*feat_test[:10,weight.index(target_norm)])
        print(genMass_test[:10])
    
        print('ggH pred',Y_pred_htt[:10])
        if target_norm!="": print('ggH pred scaled',Y_pred_htt[:10]*feat_test_htt[:10,weight_old.index(target_norm)])
    
        if target_norm!="":
            Y_pred = Y_pred*feat_test[:,weight.index(target_norm)]
            Y_pred_htt = Y_pred_htt*feat_test_htt[:,weight_old.index(target_norm)]
            if args.load:
                if args.loadspec:
                    if dtype=="hadhad":
                        Y_pred_qcd = Y_pred_qcd*feat_test_qcd[:,weight_old.index(target_norm)]
                    else:
                        if dtype=="hadel":
                            Y_pred_zee = Y_pred_zee*feat_test_zee[:,weight_old.index(target_norm)]
                        if dtype=="hadmu":
                            Y_pred_zmm = Y_pred_zmm*feat_test_zmm[:,weight_old.index(target_norm)]
                        Y_pred_wjets = Y_pred_wjets*feat_test_wjets[:,weight_old.index(target_norm)]
                    Y_pred_ttbar = Y_pred_ttbar*feat_test_ttbar[:,weight_old.index(target_norm)]
                Y_pred_ztt = Y_pred_ztt*feat_test_ztt[:,weight_old.index(target_norm)]
    
        for w in bin_dict:
            plt.clf()
            #feathist,featbins = np.histogram(feat_train[:,features_to_plot.index(w)],bins=bin_dict[w])
            #plt.hist2d(feat_train[:,features_to_plot.index(w)],(feat_train[:,weight.index('fj_msd')]-genMass_train)/genMass_train,bins=[bin_dict[w],mbins],weights=1./np.append(feathist,1.)[np.digitize(feat_train[:,features_to_plot.index(w)],featbins)-1])
            plt.hist2d(feat_train[:,features_to_plot.index(w)],(feat_train[:,weight.index('fj_msd')]-genMass_train)/genMass_train,bins=[bin_dict[w],mbins])
            plt.legend(loc='best')
            plt.xlabel(w)
            plt.ylabel('msd-mgen/mgen')
            plt.savefig("plots/massreg_2d_%s_%s.pdf"%(w,dtype))
    
            plt.clf()
            feathist,featbins = np.histogram(feat_test[:,features_to_plot.index('fj_msd')],bins=bin_dict[w])
            #plt.hist2d(feat_test[:,features_to_plot.index('fj_msd')],((Y_pred)-genMass_test)/genMass_test,bins=[bin_dict[w],mbins],weights=1./np.append(feathist,1.)[np.digitize(feat_test[:,features_to_plot.index('fj_msd')],featbins)-1])
            plt.hist2d(feat_test[:,features_to_plot.index('fj_msd')],((Y_pred)-genMass_test)/genMass_test,bins=[bin_dict[w],mbins])
            plt.legend(loc='best')
            plt.xlabel(w)
            plt.ylabel('mcorr-mgen/mgen')
            plt.savefig("plots/masscorr_2d_%s_%s.pdf"%(w,dtype))
    
        plt.clf()
        plt.hist((feat_test[:,weight.index('fj_msd')]-genMass_test)/genMass_test,bins=mbins,histtype='step',density=True,fill=False,label='SoftDrop (Flat Htt)')
        #plt.hist(((Y_pred*feat_test[:,weight.index(target_norm)])-genMass_test)/genMass_test,bins=mbins,histtype='step',density=True,fill=False,label='Regression (Flat Htt)')
        plt.hist(((Y_pred)-genMass_test)/genMass_test,bins=mbins,histtype='step',density=True,fill=False,label='Regression (Flat Htt)')
        plt.hist((feat_test_htt[:,weight.index('fj_msd')]-125.)/125.,bins=mbins,histtype='step',density=True,fill=False,label='SoftDrop (ggHtt)')
        plt.hist(((Y_pred_htt)-125.)/125.,bins=mbins,histtype='step',density=True,fill=False,label='Regression (ggHtt)')
        #plt.hist((((Y_pred_htt*feat_test_htt[:,weight.index(target_norm)])-125.)/125.,bins=mbins,histtype='step',density=True,fill=False,label='Regression (ggHtt)')
        plt.legend(loc='best')
        plt.xlabel('m-mgen/mgen')
        plt.ylabel('arb.')
        plt.title(dtype)
        plt.savefig("plots/massreg%s_%s_%s.pdf"%(full_label,'fj_msd',dtype))

        httbins = np.arange(0.,510.,10.)

        def make_med_rms(pred):
            med = np.quantile(pred,0.5)
            rms = (np.quantile(pred,0.84)-np.quantile(pred,0.16))/2.
            #deltas = (pred-mean)**2
            #sqrtdeltas = np.sqrt(deltas)
            #print(sum((sqrtdeltas>100.)),len(sqrtdeltas))
            #return med,np.sqrt(np.mean(deltas))
            return med,rms

        plt.clf()
        plt.hist((feat_test_htt[:,weight.index('fj_msd')]),bins=httbins,histtype='step',fill=False,label='SoftDrop (ggHtt : {:.2f} +- {:.2f})'.format(*(make_med_rms(feat_test_htt[:,weight.index('fj_msd')]))),density=True)
        plt.hist((Y_pred_htt),bins=httbins,histtype='step',fill=False,label='Regression (ggHtt : {:.2f} +- {:.2f})'.format(*(make_med_rms(Y_pred_htt))),density=True)
        if args.load:
            if args.loadspec:
                if dtype=="hadhad":
                    plt.hist((feat_test_qcd[:,weight.index('fj_msd')]),bins=httbins,histtype='step',fill=False,label='SoftDrop (QCD)',density=True)
                    plt.hist((Y_pred_qcd),bins=httbins,histtype='step',fill=False,label='Regression (QCD)',density=True)
                else:
                    plt.hist((feat_test_wjets[:,weight.index('fj_msd')]),bins=httbins,histtype='step',fill=False,label='SoftDrop (WJets)',density=True)
                    plt.hist((Y_pred_wjets),bins=httbins,histtype='step',fill=False,label='Regression (WJets)',density=True)
                    #if dtype=="hadel":
                        #plt.hist((feat_test_zee[:,weight.index('fj_msd')]),bins=httbins,histtype='step',fill=False,label='SoftDrop (Zee : {:.2f} +- {:.2f})'.format(*(make_med_rms(feat_test_zee[:,weight.index('fj_msd')]))),density=True)
                        #plt.hist((Y_pred_zee),bins=httbins,histtype='step',fill=False,label='Regression (Zee : {:.2f} +- {:.2f})'.format(*(make_med_rms(Y_pred_zee))),density=True)
                    #if dtype=="hadmu":
                        #plt.hist((feat_test_zmm[:,weight.index('fj_msd')]),bins=httbins,histtype='step',fill=False,label='SoftDrop (Zmm : {:.2f} +- {:.2f})'.format(*(make_med_rms(feat_test_zmm[:,weight.index('fj_msd')]))),density=True)
                        #plt.hist((Y_pred_zmm),bins=httbins,histtype='step',fill=False,label='Regression (Zmm : {:.2f} +- {:.2f})'.format(*(make_med_rms(Y_pred_zmm))),density=True)
                plt.hist((feat_test_ttbar[:,weight.index('fj_msd')]),bins=httbins,histtype='step',fill=False,label='SoftDrop (TTbar)',density=True)
                plt.hist((Y_pred_ttbar),bins=httbins,histtype='step',fill=False,label='Regression (TTbar)',density=True)
            plt.hist((feat_test_ztt[:,weight.index('fj_msd')]),bins=httbins,histtype='step',fill=False,label='SoftDrop (Ztt : {:.2f} +- {:.2f})'.format(*(make_med_rms(feat_test_ztt[:,weight.index('fj_msd')]))),density=True)
            plt.hist((Y_pred_ztt),bins=httbins,histtype='step',fill=False,label='Regression (Ztt : {:.2f} +- {:.2f})'.format(*(make_med_rms(Y_pred_ztt))),density=True)
        #plt.hist((((Y_pred_htt*feat_test_htt[:,weight.index(target_norm)])-125.)/125.,bins=mbins,histtype='step',density=True,fill=False,label='Regression (ggHtt)')
        #plt.yscale('log')
        plt.legend(loc='best')
        plt.xlabel('m')
        plt.ylabel('arb.')
        plt.title(dtype)
        plt.savefig("plots/masshtt%s_%s_%s.pdf"%(full_label,'fj_msd',dtype))

