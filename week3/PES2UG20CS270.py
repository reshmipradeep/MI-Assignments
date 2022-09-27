'''
Assume df is a pandas dataframe object of the dataset given
'''

import numpy as np
import pandas as pd

'''Calculate the entropy of the entire dataset'''
# input:pandas_dataframe
# output:int/float
def get_entropy_of_dataset(df):
    entropy=0
    if df.empty: # Return 0 if empty dataset
        return entropy
    lastColn = df.columns[-1] 
    targetval = df[lastColn].unique()

    for val in targetval:
        length = len(df[lastColn])
        pi = df[lastColn].value_counts()[val]/length
        if(pi==0):
            continue
        entropy = entropy+(-(pi*np.log2(pi)))
    return entropy


'''Return avg_info of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float
def get_avg_info_of_attribute(df, attribute):
    avg_info = 0
    if df.empty:
        return avg_info
    try:
        lastColn = df.columns[-1]
        targetval = df[lastColn].unique()
        att_val = df[attribute].unique()
        #w all the attribute value skjs ha sg adddedg to zghus  vkue 
        for val in att_val:
            entropy=0
            deno = len(df[attribute][df[attribute] == val])
            for tar_val in targetval:
                nume = len(df[attribute][df[attribute] == val][df[lastColn] == tar_val])
                pi = nume/deno
                if(pi==0):
                    continue
                entropy = entropy+(-(pi*np.log2(pi)))
            avg_info = avg_info+((deno/len(df))*entropy)

    except KeyError:
        print("Attribute ", attribute, "is not present in dataset")
    return avg_info


'''Return Information Gain of the attribute provided as parameter'''
# input:pandas_dataframe,str
# output:int/float
def get_information_gain(df, attribute):
    information_gain=0
    try:
        information_gain = get_entropy_of_dataset(df) - get_avg_info_of_attribute(df,attribute)
    except KeyError:
        print("Attribute ",attribute,"is not present in dataset")
    return information_gain


#input: pandas_dataframe
#output: ({dict},'str')
def get_selected_attribute(df):
    '''
    Return a tuple with the first element as a dictionary which has IG of all columns 
    and the second element as a string with the name of the column selected

    example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
    '''

    if df.empty or len(df.columns)==1:
        return (dict(),'')

    att=list(df)[:-1]
    att_infogain = list(map(lambda x:get_information_gain(df,x),att))
    infogain = dict(zip(att,att_infogain))

    max_index, _ =max(enumerate(att_infogain),key = lambda x:x[1]) #index of cloumn with the maximum information gain
    return (infogain,att[max_index])
