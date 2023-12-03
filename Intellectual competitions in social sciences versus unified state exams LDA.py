#!/usr/bin/env python
# coding: utf-8

# In[27]:


get_ipython().system(' pip install pyLDAvis')


# In[28]:


get_ipython().system(' pip install nltk')


# In[29]:


get_ipython().system(' pip install gensim')


# In[30]:


get_ipython().system(' pip install wordcloud')


# In[34]:


get_ipython().system(' pip install textblob')


# In[35]:


import nltk
import wordcloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from gensim import corpora, models
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from textblob import TextBlob


# In[36]:


nltk.download('stopwords')
nltk.download('punkt')


# In[37]:


def preprocess_text(text): 
    stop_words = set(stopwords.words('english'))
    extra_stop_words = {'text', 'read', 'number', 'item', 'author', 'nikolai', 'task', 'answer', 'do', 'analyze', 'complete', 'exercise', 'describe', 'write', 'give', 'choose', 'question', 'posit', 'econ', 'social', 'polit'}
    stop_words.update(extra_stop_words)
    
    ps = PorterStemmer()
    if isinstance(text, list): 
        text = ''.join(text)
    words = word_tokenize(text.lower())
    
    filtered_words = [ps.stem(word) for word in words if word.isalnum() and word not in stop_words and not any(char.isdigit() for char in word)]

    return filtered_words


# In[38]:


texts = []
file_paths = ['state_18.txt', 'state_22.txt', 'vp17_18.txt', 'vp18_19.txt', 'vp21_22.txt', 'vp22_23.txt']


# In[39]:


for file_path in file_paths: 
    with open(file_path, 'r', encoding='utf-8') as file: 
        text = file.read()
        texts.append(text)


# In[40]:


preprocessed_texts = [preprocess_text(text) for text in texts]


# In[41]:


dictionary = corpora.Dictionary(preprocessed_texts)


# In[42]:


corpus = [dictionary.doc2bow(text) for text in preprocessed_texts]


# In[43]:


lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=20)


# In[44]:


pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis, 'lda_visualization.html')


# In[ ]:


print('TOPICS:')
for topic_id, topic in lda_model.print_topics(): 
    print(f'Topic {topic_id +1}: {topic}')


# In[45]:


topic_distribution = []
for i, text in enumerate(texts):
    bow = dictionary.doc2bow(preprocess_text(text))
    topics = lda_model.get_document_topics(bow)
    total_probability = sum(prob for _, prob in topics)
    topics_dict = {f'Topic {topic + 1}': (prob / total_probability) * 100 for topic, prob in topics}
    
    topic_distribution.append({'Document': f'Document {i + 1}', **topics_dict})


df = pd.DataFrame(topic_distribution).set_index('Document').fillna(0)  # Fill NaN values with 0

print(df)


# In[46]:


pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(vis)


# In[47]:


vis_data = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis_data, 'topic_modeling_visualization.html')


# In[49]:



print("Topics:")
for topic_id, topic in lda_model.print_topics():
    print(f"Topic {topic_id + 1}: {topic}")


for i, text in enumerate(texts):
    bow = dictionary.doc2bow(preprocess_text(text))
    topics = lda_model.get_document_topics(bow)
    print(f"Document {i + 1} Topics: {topics}")

    
    for topic_id, topic_prob in topics:
        topic_words = lda_model.show_topic(topic_id, topn=10)  # Adjust the number of words in the word cloud
        wordcloud = WordCloud(width=400, height=200, background_color='white').generate_from_frequencies(dict(topic_words))

        
        plt.figure(figsize=(6, 3))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Topic {topic_id + 1} Word Cloud for Document {i + 1}')
        plt.show()

