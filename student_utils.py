import pandas as pd
import numpy as np
import os
import tensorflow as tf

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    df = pd.merge(df, ndc_df[['Non-proprietary Name', 'NDC_Code']],
                  left_on='ndc_code', 
                  right_on='NDC_Code')
    df['generic_drug_name'] = df['Non-proprietary Name']
    return df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    first_encounter_df=df.sort_values(['encounter_id'], ascending=True).groupby('patient_nbr').head(1)
    
    #df = df.sort_values(encounter_id)
    #first_encounter_values = df.groupby(patient_id)[encounter_id].head(1).values
    #first_encounter_df=df[df[encounter_id].isin(first_encounter_values)] 
    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    df = df.iloc[np.random.permutation(len(df))]
    unique_values = df[patient_key].unique()
    total_values = len(unique_values)
    
    train_size = round(total_values * (0.6))
    vt_size = round(total_values * (0.2)) + train_size
    
    train = df[df[patient_key].isin(unique_values[:train_size])].reset_index(drop=True)
    validation = df[df[patient_key].isin(unique_values[train_size:vt_size])].reset_index(drop=True)
    test = df[df[patient_key].isin(unique_values[vt_size:])].reset_index(drop=True)
    
    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        tf_categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(key = c, vocabulary_file = vocab_file_path, num_oov_buckets=0)       
        if c == 'primary_diagnosis_code':
            dims = 10
            cat_col = tf.feature_column.embedding_column(tf_categorical_feature_column, dimension=dims)        
        else:
            cat_col = tf.feature_column.indicator_column(tf_categorical_feature_column)      
        output_tf_list.append(cat_col)
        
        
    return output_tf_list


#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = '?'
    s = '?'
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    return student_binary_prediction
