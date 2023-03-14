import streamlit as st
import requests
import openai
import pandas as pd
'''
# Welcome to Mind Magic
'''
# 1. Get the text to be tested

prediction_url = 'Prediction URL' #URL of AI Powerered language testing API
news_api_url='https://newsdata.io/api/1/news' # URL of the api generating news articles from a topic/keyword
openai.api_key='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'

# Asking the user to ask the source of the text
text_source = st.radio('Where from do you want to fetch your text?', ('Enter a text', 'News article', 'Ask ChatGPT'))

# >>>>>>>>1st Scenario : the user enters a text
if text_source == 'Enter a text':
    #st.write('Enter a text')
    full_text = st.text_input('Enter a text','')

# >>>>>>>>2nd Scenario : the user enters a topic on the basis of which the news article will be fetched
elif text_source == 'News article':
    #st.write('Enter topic/keyword(s)')
    keyword_topic = st.text_input('Enter topic/keyword(s)', '')
    query_params = {
    'apikey':'pub_18816cd4733fe56882fecbaede780cd02a6bc',
    'q':keyword_topic,
    'country':'us',
    'language':'en',
    'category':'business',
}
    if keyword_topic:
        res = requests.get(news_api_url, params=query_params)
        full_text= res.json()['results'][0]["content"]
        st.write(full_text)

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
    full_text=chatgpt_response['choices'][0]['message']['content']
    if gpt_question:
        st.write(full_text)

# st.markdown('''
# Remember that there are several ways to output content into your web page...

# Either as with the title by just creating a string (or an f-string). Or as with this paragraph using the `st.` functions
# ''')

#2. Dictionary of the parameters for our API...
#params=dict(full_text=full_text)

# 3. Let's call our API using the `requests` package...
#res=requests.get(url, params).json()

# 4. Let's retrieve the prediction from the **JSON** returned by the API...
#scores=res['evaluation_score']

scores={"cohesion" : [3], "syntax": [3] , "vocabulary": [3], "phraseology": [3], "grammar": [3], "conventions": [3] }
# 5. we can display the prediction to the user
scores_table= pd.DataFrame.from_dict(scores)
scores_table
