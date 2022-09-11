
import pandas as pd
import numpy as np

import pickle
import streamlit as st
import tensorflow as tf
from joblib import dump, load



def preprocess_data(df):
    dfn = df.copy()
    dfc = dfn.copy()
    top_c = ['BRA', 'PRT', 'FRA', 'DEU', 'ITA', 'GBR', 'ESP', 'USA', 'NLD', 'CHE']
    dfc["new_nationality"] = dfc["Nationality"].apply(lambda x: x if x in top_c else "Other")
    drop_cols = ['Nationality',]
    cat_cols = ["new_nationality","MarketSegment","DistributionChannel"]
    
    ohe = load('tools/encoder.joblib')
    sc = load('tools/scaler.joblib')
    
    encoded_train = ohe.transform(dfc[cat_cols]).toarray()
    
    ohdf = pd.DataFrame(encoded_train,columns = ohe.get_feature_names(cat_cols))
    
    combines_frames = [dfc,ohdf]
    dfc = pd.concat(combines_frames,axis=1)

    dfc = dfc.drop(drop_cols,axis=1)
    dfc = dfc.drop(cat_cols,axis=1)
    
    X = dfc

    X = sc.transform(X)
    
    return X

def predict(data):
    
    cols = ['Nationality', 'Age', 'DaysSinceCreation',
       'AverageLeadTime', 'LodgingRevenue', 'OtherRevenue', 'PersonsNights', 'RoomNights',
       'DaysSinceLastStay', 'DaysSinceFirstStay', 'DistributionChannel',
       'MarketSegment', 'SRHighFloor', 'SRLowFloor', 'SRAccessibleRoom',
       'SRMediumFloor', 'SRBathtub', 'SRShower', 'SRCrib', 'SRKingSizeBed',
       'SRTwinBed', 'SRNearElevator', 'SRAwayFromElevator',
       'SRNoAlcoholInMiniBar', 'SRQuietRoom']

    df = pd.DataFrame([data],columns=cols)
    X = preprocess_data(df)

    model = tf.keras.models.load_model('nn/')
    results = model.predict(X)
    pred_val = [x.argmax() for x in results][0]
    return pred_val
    
  
def main():
    
    # giving a title
    st.title('Hotel Customer check in classification app')
    
    # getting the input data from the user
    n1 = st.selectbox('Customer Nationality',
     ('BRA', 'PRT', 'FRA', 'DEU', 'ITA', 'GBR', 'ESP', 'USA', 'NLD', 'CHE', 'Other'))
    n2 = st.text_input('Age')
    n3 = st.text_input('Days since creation')
    n4 = st.text_input('Average Lead time')
    n5 = st.text_input('Lodging Revenue')
    n6 = st.text_input('Other revenue')
    n7 = st.text_input('Persons Nights')
    n8 = st.text_input('Room nights')
    n9 = st.text_input('Days since last stay')
    n10 = st.text_input('Days since first stay')
    n11 = st.selectbox('Distribution Channel',
     ('Corporate', 'Direct', 'Electronic Distribution', 'Travel Agent/Operator',))
    n12 = st.selectbox('Market Segment',
     ('Aviation', 'Complementary', 'Corporate', 'Direct', 'Other', 'Travel Agent/Operator'))
    n13 = st.text_input('SR High Floor')
    n14 = st.text_input('SR Low Floor')
    n15 = st.text_input('SR Accessible room')
    n16 = st.text_input('SR Medium Floor')
    n17 = st.text_input('SR Bathtub')
    n18 = st.text_input('SR Shower')
    n19 = st.text_input('SR Crib')
    n20 = st.text_input('SR King size bed')
    n21 = st.text_input('SR twin bed')
    n22 = st.text_input('SR Near elevator')
    n23 = st.text_input('SR away from elevator')
    n24 = st.text_input('SR no alcohol mini bar')
    n25 = st.text_input('SR quiet room')

    result = ''
    
    if st.button('get prediction'):
        val = predict([n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n16,n17,n18,n19,n20,n21,n22,n23,n24,n25])

        if val == 0:
            result = 'customer will most likely check in'
        else:
            result = 'customer will most likely NOT check in'
        
    st.success(result)

    
if __name__ == '__main__':
    main()