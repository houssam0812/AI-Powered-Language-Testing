import streamlit as st
import requests
import openai
import pandas as pd

'''
# Welcome to Mind Magic
'''
# 0. Fetching API keys from secrets
spell = st.secrets['spell']
news_api_key = st.secrets.news_api.key
openai_api_key= st.secrets.openai_api.key
openai.api_key=openai_api_key

# 1. Get the text to be tested
prediction_url = 'Prediction URL' #URL of AI Powerered language testing API
news_api_url='https://newsdata.io/api/1/news' # URL of the api generating news articles from a topic/keyword


# Asking the user to ask the source of the text
text_source = st.radio('Where from do you want to fetch your text?', ('Enter a text', 'News article', 'Ask ChatGPT'))
full_text=''
# >>>>>>>>1st Scenario : the user enters a text
if text_source == 'Enter a text':
    #st.write('Enter a text')
    full_text = st.text_input('Enter a text','')

# >>>>>>>>2nd Scenario : the user enters a topic on the basis of which the news article will be fetched
elif text_source == 'News article':
    #st.write('Enter topic/keyword(s)')
    keyword_topic = st.text_input('Enter topic/keyword(s)', '')
    query_params = {
    'apikey':news_api_key,
    'q':keyword_topic,
    'country':'us',
    'language':'en',
    'category':'business',
}
    if keyword_topic:
        res = requests.get(news_api_url, params=query_params)
        full_text= res.json()['results'][0]["content"]
        st.markdown(f'<div style="text-align: justify;">{full_text}</div>', unsafe_allow_html=True)
# >>>>>>>>3rd Scenario : the user asks chatgpt for a text based
else:
    #st.write('Ask ChatGPT')
    gpt_question = st.text_input('Enter your question here', '')
    chatgpt_response=openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "Hello, can you help me write a text ?"},
        {"role": "user", "content": gpt_question},

    ]
  )
    if gpt_question:
        full_text=chatgpt_response['choices'][0]['message']['content']

        #st.write(full_text)
        st.markdown(f'<div style="text-align: justify;">{full_text}</div>', unsafe_allow_html=True)
#2. Dictionary of the parameters for our API...
#params=dict(full_text=full_text)

# 3. Let's call our API using the `requests` package...
#res=requests.get(url, params).json()

# 4. Let's retrieve the prediction from the **JSON** returned by the API...
#scores=res['evaluation_score']

scores={"cohesion" : [2.5574], "syntax": [1] , "vocabulary": [3], "phraseology": [3], "grammar": [3], "conventions": [3] }
# 5. we can display the prediction to the user
scores_table= pd.DataFrame.from_dict(scores)

columns = st.columns(6)
if full_text:
    for i in range(len(scores_table.columns)):
        #score_type = columns[i].write(scores_table.columns[i])
        score_type = columns[i].markdown(f'<div style="text-align: center;">{scores_table.columns[i]}</div>', unsafe_allow_html=True)
        if scores_table.loc[0,scores_table.columns[i]]>=2.5:
            score_value=columns[i].success(scores_table.loc[0,scores_table.columns[i]])
        else:
            score_value=columns[i].error(scores_table.loc[0,scores_table.columns[i]])
