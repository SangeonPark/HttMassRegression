import os,sys
#import tensorflow as tf
import numpy as np
from optparse import OptionParser
import argparse
import pandas as pd
import h5py
import json
from keras.models import model_from_json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--tag',        dest='tag',        default="",           help="tag")
    parser.add_argument('--label',      dest='label',      default="",           help="label")
    parser.add_argument('--out',        dest='out',        default="",        help='out')
    args = parser.parse_args()

    dtype = args.tag
    full_label = args.label
    if full_label!="":
        full_label = "_"+full_label
    full_out = args.out
    if full_out!="":
        full_out = "_"+full_out
    
    models = {}
    for m in ['regression']:
        os.system('cp models/model'+full_label+'_'+dtype+'_'+str(m)+'.json models/fullmodel_'+dtype+'_'+str(m)+(full_label if full_out=='' else full_out)+'.json')
        os.system('cp models/model'+full_label+'_'+dtype+'_'+str(m)+'.h5 models/fullmodel_'+dtype+'_'+str(m)+(full_label if full_out=='' else full_out)+'_weights.h5')
        json_file = open('models/model'+full_label+'_'+dtype+'_'+str(m)+'.json', 'r')
        model_json = json_file.read()
        models[m] = model_from_json(model_json)
        models[m].load_weights("models/model"+full_label+"_"+dtype+"_"+str(m)+".h5")
        print(str(m), models[m].summary())
        models[m].save("models/fullmodel_"+dtype+"_"+str(m)+(full_label if full_out=="" else full_out)+".h5")
