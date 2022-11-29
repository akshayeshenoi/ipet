import numpy as np
import os 
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Concatenate, Input, Dropout,Conv1D, GRU, Concatenate,MaxPooling1D,Input, Dropout,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from constants import train_sit


def custom_split(features_main,silences,labels):
    idx_new = np.random.permutation(features_main.shape[0])
    features_main,silences,labels = features_main[idx_new], silences[idx_new], labels[idx_new]
    
    split = int(len(features_main)*0.7)
    X_train_feat = features_main[:split]
    X_test_feat = features_main[split:]

    X_train_silence = silences[:split]
    X_test_silence = silences[split:]

    Y_train = labels[:split]
    Y_test = labels[split:]

    return X_train_feat,X_test_feat,X_train_silence,X_test_silence,Y_train,Y_test

def LSTM_discriminator(x_sample,y_sample):
    dtype = 'float64'
    dropout=0.2

    input_data_1 = Input(name='Main_Features', shape = x_sample[1:], dtype=dtype)
    input_data_2 = Input(name='Silence', shape=(1), dtype=dtype)

    att_in = Bidirectional(LSTM(8,return_sequences=True,kernel_regularizer=l2()))(input_data_1)
    att_out = Bidirectional(LSTM(8,return_sequences=False))(att_in)
    
    concatted = Concatenate()([att_out,input_data_2])
    x = Dense(units=128, activation='tanh', name='fc')(concatted)
    x = Dropout(dropout, name='dropout_2')(x)

    # Output layer with softmax
    y_pred = Dense(units=y_sample, activation='softmax', name='softmax')(x) 

    K.clear_session()
    model = Model(inputs=[input_data_1,input_data_2], outputs=y_pred)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model 

def CNN_discriminator(x_sample,y_sample):
    dtype = 'float64'
    dropout=0.2
    
    n_timesteps, n_features = x_sample[1], x_sample[2]

    input_data_1 = Input(name='Main_Features', shape = (x_sample[1],x_sample[2]), dtype=dtype)
    input_data_2 = Input(name='Silence', shape=(1), dtype=dtype)

    att_in = Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features))(input_data_1)
    att_out = Conv1D(filters=64, kernel_size=3, activation='relu')(att_in)
    x = Dropout(0.5)(att_out)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    
    concatted = Concatenate()([x,input_data_2])
    x = Dense(units=100, activation='tanh', name='fc')(concatted)

    y_pred = Dense(units=y_sample, activation='softmax', name='softmax')(x) 

    K.clear_session()
    model = Model(inputs=[input_data_1,input_data_2], outputs=y_pred)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model 

def GRU_discriminator(x_sample,y_sample):
    dtype = 'float64'
    dropout=0.2

    input_data_1 = Input(name='Main_Features', shape = (x_sample[1],x_sample[2]), dtype=dtype)
    input_data_2 = Input(name='Silence', shape=(1), dtype=dtype)

    att_in = GRU(32,return_sequences=True,kernel_regularizer=l2())(input_data_1)
    att_out = GRU(32,return_sequences=False)(att_in)
    
    concatted = Concatenate()([att_out,input_data_2])
    x = Dense(units=128, activation='tanh', name='fc')(concatted)
    x = Dropout(dropout, name='dropout_2')(x)

    # Output layer with softmax
    y_pred = Dense(units=y_sample, activation='softmax', name='softmax')(x) 

    K.clear_session()
    model = Model(inputs=[input_data_1,input_data_2], outputs=y_pred)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model 

def MLP_discriminator(x_sample,y_sample):
    dtype = 'float64'
    dropout=0.2
    
    input_data_1 = Input(name='Main_Features', shape = (x_sample[1],x_sample[2]), dtype=dtype)
    input_data_2 = Input(name='Silence', shape=(1), dtype=dtype)

    x = Dense(units=128, activation='tanh', name='fc1')(input_data_1)
    x = Dropout(dropout, name='dropout_2')(x)
    x = Dense(units=32, activation='tanh', name='fc2')(x)
    x = Flatten()(x)

    concatted = Concatenate()([x,input_data_2])
    x = Dense(units=100, activation='tanh', name='fc3')(concatted)
    x = Dropout(dropout, name='dropout_3')(x)

    # Output layer with softmax
    y_pred = Dense(units=y_sample, activation='softmax', name='softmax')(x) 

    K.clear_session()
    model = Model(inputs=[input_data_1,input_data_2], outputs=y_pred)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model 


def get_model():
    x_shape = X_train_feat.shape
    y_shape = Y_test.shape[1]
    if train_sit == 'LSTM':
        return LSTM_discriminator(x_shape,y_shape)
    elif train_sit == 'CNN':
        return CNN_discriminator(x_shape,y_shape)
    elif train_sit == 'GRU':
        return GRU_discriminator(x_shape,y_shape)
    elif train_sit == 'MLP':
        return MLP_discriminator(x_shape,y_shape)
    else:
        raise Exception("Sorry, invalid Attack Model")

if __name__ == '__main__':
    print("Loading Features.......")
    combined_main_features = np.load('data/Traces_Mseq/new_features_main.npy')
    combined_silences = np.load('data/Traces_Mseq//new_silences.npy')
    combined_labels = np.load('data/Traces_Mseq//new_labels.npy')

    print("Preparing Feature Vectors.......")
    ohe = OneHotEncoder(sparse=False)
    labels_u = combined_labels.reshape(-1,1)
    labels_ohe = ohe.fit_transform(labels_u)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    K.clear_session()
    np.random.seed(20)
    
    X_train_feat,X_test_feat,X_train_silence,X_test_silence,Y_train,Y_test = custom_split(combined_main_features,combined_silences,labels_ohe)

    adam = Adam()
    early = EarlyStopping(monitor="val_loss",
                                mode="min",
                                patience=10)
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                            patience=3,
                                            verbose=1, mode='min', min_delta=0.001, cooldown=3,
                                            min_lr=1e-9) 

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
    tf.keras.backend.set_floatx('float64')   

    checkpoint = ModelCheckpoint('Models/M_seq_fingeprinting', monitor='val_loss', verbose=0,
                                    save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss",
                                mode="min",
                                patience=16)
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.7,
                                            patience=5,
                                            verbose=1, mode='min', min_delta=0.0001, cooldown=3,
                                            min_lr=1e-9)  
    
    model = get_model()
    model.fit([X_train_feat,X_train_silence], Y_train,
                        batch_size=128, epochs=100,
                        validation_data=([X_test_feat,X_test_silence], Y_test),
                        callbacks=[early, reduceLROnPlat,checkpoint])