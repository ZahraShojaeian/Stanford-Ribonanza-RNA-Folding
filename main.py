import argparse
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import numpy as np
import pandas as pd
from Model import data_process, train
import joblib

def main(config):
    mode = config.mode
    n_s = config.n_s
    num_epoch = config.num_epoch
    batch_size = config.batch_size
    train_data_path = config.train_data_path
    train_data = pd.read_csv(train_data_path)
    if mode == "train":
        TR = train(config)
        DT = data_process(config, train_data)
        one_hot_X, Y = DT.prepare_data()
        X_train, X_val, Y_train, Y_val = train_test_split(one_hot_X, Y, test_size=0.1, random_state=42)
        early_stopping = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
        model = TR.modelf()
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer= opt, loss='mean_absolute_error')
        s0_train = np.zeros((len(X_train), n_s))
        c0_train = np.zeros((len(X_train), n_s))
        s0_val = np.zeros((len(X_val), n_s))
        c0_val = np.zeros((len(X_val), n_s))
        
        model.fit([X_train, s0_train, c0_train], Y_train, epochs=num_epoch, batch_size=batch_size, 
                  validation_data=([X_val, s0_val, c0_val], Y_val), callbacks=[early_stopping])
        
        joblib.dump(model, 'output_model.joblib')
        

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=['train', 'test'])
    ap.add_argument("--train_data_path",default="./train_data.csv")
    ap.add_argument("--test_data_path",default="./test_sequences.csv")
    ap.add_argument("--Tx", type=int,default=457)
    ap.add_argument("--n_a", type=int,default=128)
    ap.add_argument("--n_s", type=int,default=256)
    ap.add_argument("--input_size", type=int,default=4)
    ap.add_argument("--num_epoch", type=int,default=1000)
    ap.add_argument("--batch_size", type=int,default=256)
    ap.add_argument("--experiment_type",type=str, choices=['2A3_MaP', 'DMS_MaP'])
    config = ap.parse_args()
    main(config)