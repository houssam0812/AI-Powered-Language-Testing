import streamlit as st
import requests

'''
# Welcome to Mind Magic
'''

text_source = st.radio('Select a data source', ('Enter a text', 'News article', 'Ask ChatGPT'))
st.write(text_source)

if text_source == 'Enter a text':
    full_text = st.text_input('Enter a text')
elif text_source == 'News article':
    full_text = st.text_input('Enter a URL')

elif text_source == 'Ask ChatGPT':
    full_text = st.text_input('Enter a URL')
else:
    st.write('◀️')
# st.markdown('''
# Remember that there are several ways to output content into your web page...

# Either as with the title by just creating a string (or an f-string). Or as with this paragraph using the `st.` functions
# ''')

# '''
# ## Here we would like to add some controllers in order to ask the user to select the parameters of the ride

# 1. Let's ask for:
# - date and time
# - pickup longitude
# - pickup latitude
# - dropoff longitude
# - dropoff latitude
# - passenger count
# '''
full_text = st.text_input('Enter a text, a URL or ask ChatGPT to generate a text')
# st.write(d, str(t))


url = 'https://taxifare.lewagon.ai/predict'

# 2. Let's build a dictionary containing the parameters for our API...
params=dict(full_text=full_text
            )

# 3. Let's call our API using the `requests` package...
res=requests.get(url, params).json()

# 4. Let's retrieve the prediction from the **JSON** returned by the API...
scores=res['evaluation_score']

# 5. we can display the prediction to the user
st.write("FARE:",fare)
