#!/usr/bin/env python3

"""
Script to perform cross validation on the 35 patient training set. The major advantage of this script is that it is designed to perform
patient-level cross validation, not patch or mosaic level cross validaiton.
"""

from pandas import DataFrame, ExcelWriter
import os
import numpy as np 

from sklearn.utils import shuffle
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score

import random
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.models import Model
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras import backend as K


class Kfold_split_data(object):
    """
    Object that holds the cross validation data and results
    """
    def __init__(self, train_patients, val_patients):
        self.train_patients = train_patients
        self.val_patients = val_patients
        self.labels = None
        self.filenames = None
        
        self.softmax_vector = None
        self.val_predictions = None
        
        self.train_history = None
        self.val_acc = None
        
    def set_labels(self, val_gen):
        self.labels = val_gen.classes
    def set_filenames(self, val_gen):
        self.filenames = val_gen.filenames

    def set_softmaxes(self, preds):
        self.softmax_vector = preds
    def set_val_predictions(self, preds):
        self.val_predictions = np.argmax(preds, axis = 1)

    def set_train_history(self, history):
        self.train_history = history
    def val_accuracy(self, val_gen):
        index_validation = val_gen.classes
        self.val_acc = accuracy_score(index_validation, self.val_predictions)

def file_splitter(file):
    """
    Function to return the NIO patient number
    """
    nio, _ = file.split("-")
    return nio

def nio_preprocessing_function(image):
    """
    Channel-wise means calculated over NIO dataset
    """
    image[:,:,0] -= 102.1
    image[:,:,1] -= 91.0
    image[:,:,2] -= 101.5
    return image

def parent_dataframe(directory):
    """
    Returns a dataframe with full list of files for training and validation to be cross-folded and patient list
    """
    filelist = []
    patients = []
    labels = []
     
    for root, _, files in os.walk(directory):
        for file in files:
            if "png" in file:
                filelist.append(os.path.join(root, file))
                patients.append(file_splitter(file))
                if "recurrence" in os.path.join(root, file):
                    labels.append("recurrence")
                if "pseudoprogression" in os.path.join(root, file):
                    labels.append("pseudoprogression")
                if "nondiagnostic" in os.path.join(root, file):
                    labels.append("nondiagnostic")
    
    df = DataFrame({"filename":filelist, "class":labels})
    df = shuffle(df) # shuffle dataframe

    # filter list into unique set of patients/NIO numbers
    patients = sorted(list(set(patients)))
    SEED = 6
    random.seed(SEED)
    random.shuffle(patients)
    return patients, df

def model_builder(parent_model, trainable_feature_extractor = True):
    """
    Function that builds the recurrence model from a previously trained SRH feature extractor. 
    Can specify if the bottom CNN layers are trainable. 
    if yes > "pretraining"
    if no > "transfer training"

    """
    # define feature extractor model    
    feature_extractor = Model(input=parent_model.inputs, outputs=parent_model.layers[-7].output) # indexing into the global average pooling layer, -7
    
    if trainable_feature_extractor:
        feature_extractor.trainable = True
    elif not trainable_feature_extractor:
        feature_extractor.trainable = False

    # rebuild last layers
    x = feature_extractor.output
    x = Dense(TOTAL_CLASSES, kernel_initializer='he_normal')(x)
    predictions = Activation('softmax', name='srh_activation_2')(x)
    model = Model(inputs=feature_extractor.input, outputs=predictions)

    return model

def kfold_generator(df, patient_list, kth_fold):
    """
    Generates folds for cross validation
    """
    val_set = patient_list[kth_fold * 5:kth_fold * 5 + 5] # THIS IS SPECIFIC FOR 35 and would need to change for different number of patients
    train_set = list(set(patient_list).difference(val_set))
    
    # generate training dataframe
    train_df = DataFrame()
    for train_case in train_set:
        filter_df = df[df['filename'].str.contains(train_case)]
        train_df = train_df.append(filter_df, ignore_index=True)

    # generate validation dataframe
    val_df = DataFrame()
    for val_case in val_set:
        filter_df = df[df['filename'].str.contains(val_case)]
        val_df = val_df.append(filter_df, ignore_index=True)

    return train_set, val_set, train_df, val_df

def class_weights(train_generator):
    weights = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes), train_generator.classes)
    weight_dict = dict(zip(list(range(0,TOTAL_CLASSES)), weights))
    return weight_dict

def export_history(kfold_object, kthfold):

    # Create some Pandas dataframes from some data.
    df1 = DataFrame({'filenames': kfold_object.filenames,
    'nondiagnostic': kfold_object.softmax_vector[:,0],
    'pseudoprogression': kfold_object.softmax_vector[:,1],
    'recurrence': kfold_object.softmax_vector[:,2],
    'pred': kfold_object.val_predictions,
    'ground': kfold_object.labels})

    df2 = DataFrame({'train_patients': kfold_object.train_patients})
    df3 = DataFrame({'val_patinets': kfold_object.val_patients})

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = ExcelWriter('kthfold_results_' + str(kthfold) + ".xlsx" , engine='xlsxwriter')

    # Write each dataframe to a different worksheet.
    df1.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer, sheet_name='Sheet2')
    df3.to_excel(writer, sheet_name='Sheet3')

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()

def kfold_iterator(parent_df, patient_list, model_path):

    training_dict = {}
    for kth_fold in range(7): # this number is specific to the number of cases we have (35 total, 7 folds, 5 each)
        print("Executing", str(kth_fold) + "th fold for training.")

        # generate training and validation patients
        train_patients, val_patients, train_df, val_df = kfold_generator(parent_df, patient_list, kth_fold)
        print("Validation patients:", val_patients)

        # instantiate kfold oject
        kfold_train_object = Kfold_split_data(train_patients, val_patients)

        # initialize generators
        training_generator = ImageDataGenerator(horizontal_flip=True, 
                                                vertical_flip=True, 
                                                preprocessing_function=nio_preprocessing_function).flow_from_dataframe(
                                                        dataframe = train_df,
                                                        target_size = (IMG_ROWS, IMG_COLS),
                                                        batch_size=32, 
                                                        shuffle=True)
        
        validation_generator = ImageDataGenerator(horizontal_flip=False, 
                                                vertical_flip=False, 
                                                preprocessing_function=nio_preprocessing_function).flow_from_dataframe(
                                                        dataframe = val_df,
                                                        target_size = (IMG_ROWS, IMG_COLS),
                                                        batch_size=1,
                                                        shuffle=False)   

        # import and build model
        parent_model = load_model(model_path)
        model = model_builder(parent_model)

        # parrellel model, optimizer and compile
        parallel_model = multi_gpu_model(model, gpus=2)
        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        parallel_model.compile(optimizer=adam, loss="categorical_crossentropy", metrics =['accuracy'])

        # fit model
        history = parallel_model.fit_generator(training_generator, steps_per_epoch = STEPS_PER_EPOCH, epochs=NUM_EPOCHS, shuffle=True, class_weight = class_weights(training_generator),
                                    max_queue_size=30, workers=1, initial_epoch=0, verbose = 1)
        
        # validation_data=validation_generator, validation_steps=val_df.shape[0]
        cnn_predictions = parallel_model.predict_generator(validation_generator, steps=val_df.shape[0], verbose=1) 

        # assign object attributes after training
        kfold_train_object.set_labels(validation_generator)
        kfold_train_object.set_filenames(validation_generator)

        kfold_train_object.set_softmaxes(cnn_predictions)
        kfold_train_object.set_val_predictions(cnn_predictions)

        kfold_train_object.set_train_history(history)
        kfold_train_object.val_accuracy(validation_generator)

        print(str(kth_fold), "fold validation accuracy:", str(kfold_train_object.val_acc))
        
        # add to dictionary
        training_dict["train_epoch_" + str(kth_fold)] = kfold_train_object
        
        # export results to xlsx file
        export_history(kfold_train_object, kth_fold)

        # save trained model
        model.save("recurmodel_kthfold_" + str(kth_fold) + ".hdf5")

        # clear tensorflow session and delete models
        K.clear_session()
        del parallel_model
        del model

if __name__ == "__main__":
    
    # training specifications
    IMG_ROWS, IMG_COLS = 300, 300
    IMG_CHANNELS = 3

    TOTAL_CLASSES = 3
    
    STEPS_PER_EPOCH = 5000
    NUM_EPOCHS = 3

    # load parent dataframe
    patients, parent_df = parent_dataframe("/home/todd/Desktop/RecPseudo_data/patches")

    # iterator over the kfolds
    training_dictionary = kfold_iterator(parent_df, patients, model_path="/home/todd/Desktop/Models/model_for_activations_Final_Resnet_weights.03-0.86.hdf5")


