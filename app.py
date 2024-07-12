import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import streamlit as st


# sklearn.set_config(transform_output="pandas")

pipeline = joblib.load('Pipeline.pkl')

st.title('House Prices - Kaggle')
st.caption('Team Kostya and Nastya')
st.divider()

file = st.file_uploader(label='Загрузи данные без цены недвиги', type='csv')

if file:
    df = pd.read_csv(file)
   
    answer = np.exp(Pipeline.predict(df))
    result_df = pd.DataFrame({
        'Id': df['Id'],
        'SalePrice': answer
    })
    
    st.download_button(label='Тут прогнозы', data=result_df.to_csv(index=False), file_name='submission.csv')



# with st.sidebar:
#     st.write('')
#     file = st.file_uploader(label='Загрузи данные без цены недвиги', type='csv')
#     if file:
#         df = pd.read_csv(file)
#         answer = np.exp(Pipeline.predict(df))
#         result = pd.DataFrame({
#             'Id': df['Id'],
#             'SalePrice': answer
#         })

#         st.download_button(label='Тут прогнозы', data=result_df.to_csv(index=False), file_name='submission.csv')