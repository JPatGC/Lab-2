#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# https://www.gutenberg.org/cache/epub/11/pg11.txt


# In[5]:


import numpy as np


# In[6]:


import re


# In[7]:


import nltk


# In[4]:


from nltk import word_tokenize


# In[8]:


from nltk import sent_tokenize


# In[9]:


from nltk.util import bigrams


# In[10]:


from nltk.lm.preprocessing import padded_everygram_pipeline


# In[11]:


import requests


# In[12]:


r = requests.get(r'https://www.gutenberg.org/cache/epub/64317/pg64317.txt')
great_gatsby = r.text


# In[13]:


for char in ["\n", "\r", "\d", "\t"]:
    great_gatsby = great_gatsby.replace(char, " ")

# check
print(great_gatsby[:100])


# In[14]:


# remove the metadata at the beginning - this is slightly different for each book
great_gatsby = great_gatsby[983:]


# In[15]:


# 2 is for bigrams
n = 2
#specify the text you want to use
text = great_gatsby


# In[16]:


# step 1: tokenize the text into sentences
sentences = nltk.sent_tokenize(text)


# In[17]:


# step 2: tokenize each sentence into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# step 3: convert each word to lowercase
tokenized_text = [[word.lower() for word in sent] for sent in tokenized_sentences]

#notice the sentence breaks and what the first 10 items of the tokenized text
print(tokenized_text[0])


# In[18]:


# notice what the first 10 items are of the vocabulary
print(text[:10])


# In[19]:


# we imported this function from nltk
train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)


# In[20]:


from nltk.lm import MLE
# we imported this function from nltk linear models (lm) 
# it is for Maximum Likelihood Estimation

# MLE is the model we will use
lm = MLE(n)


# In[21]:


# currently the vocab length is 0: it has no prior knowledge
len(lm.vocab)


# In[22]:


# fit the model 
# training data is the bigrams and unigrams 
# the vocab is all the sentence tokens in the corpus 

lm.fit(train_data, padded_sents)
len(lm.vocab)


# In[23]:


# inspect the model's vocabulary. 
# be sure that a sentence you know exists (from tokenized_text) is in the 
print(lm.vocab.lookup(tokenized_text[0]))


# In[24]:


# see what happens when we include a word that is not in the vocab. 
print(lm.vocab.lookup('then wear the gold hat iphone .'.split()))


# In[25]:


# how many times does daisy appear in the model?
print(lm.counts['daisy'])

# what is the probability of daisy appearing? 
# this is technically the relative frequency of daisy appearing 
lm.score('daisy')


# In[26]:


# what is the score of 'UNK'? 

lm.score("<UNK>")


# In[27]:


# generate a 20 word sentence starting with the word, 'daisy'

print(lm.generate(20, text_seed= 'daisy', random_seed=42))


# In[28]:


from nltk.tokenize.treebank import TreebankWordDetokenizer

detokenize = TreebankWordDetokenizer().detokenize

def generate_sent(lm, num_words, text_seed, random_seed=42):
    """
    :param model: An ngram language model from `nltk.lm.model`.
    :param num_words: Max no. of words to generate.
    :param random_seed: Seed value for random.
    """
    content = []
    for token in lm.generate(num_words, text_seed=text_seed, random_seed=random_seed):
        if token == '<s>':
            continue
        if token == '</s>':
            break
        content.append(token)
    return detokenize(content)


# In[29]:


# Now generate sentences that look much nicer. 
generate_sent(lm, 20, text_seed='daisy', random_seed = 42)


# In[30]:


# Now generate sentences that look much nicer. 
generate_sent(lm, 20, text_seed='gatsby', random_seed = 62)


# In[31]:


# Now generate sentences that look much nicer. 
generate_sent(lm, 40, text_seed='tom', random_seed = 62)


# In[32]:


# Now generate sentences that look much nicer. 
generate_sent(lm, 60, text_seed='nick', random_seed = 82)


# In[33]:


r = requests.get(r'https://www.gutenberg.org/cache/epub/11/pg11.txt')
alice_adventure = r.text
# https://www.gutenberg.org/cache/epub/11/pg11.txt


# In[34]:


for char in ["\n", "\r", "\d", "\t"]:
    alice_adventure = alice_adventure.replace(char, " ")

# check
print(alice_adventure[:100])


# In[35]:


# remove the metadata at the beginning - this is slightly different for each book
alice_adventure = alice_adventure[983:]


# In[36]:


# 2 is for bigrams
n = 2
#specify the text you want to use
text = alice_adventure


# In[37]:


# step 1: tokenize the text into sentences
sentences = nltk.sent_tokenize(text)


# In[38]:


# step 2: tokenize each sentence into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# step 3: convert each word to lowercase
tokenized_text = [[word.lower() for word in sent] for sent in tokenized_sentences]

#notice the sentence breaks and what the first 10 items of the tokenized text
print(tokenized_text[0])


# In[39]:


# notice what the first 100 items are of the vocabulary
print(text[:100])


# In[40]:


# we imported this function from nltk
train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)


# In[41]:


from nltk.lm import MLE
# we imported this function from nltk linear models (lm) 
# it is for Maximum Likelihood Estimation

# MLE is the model we will use
lm = MLE(n)


# In[42]:


# currently the vocab length is 0: it has no prior knowledge
len(lm.vocab)


# In[43]:


# fit the model 
# training data is the bigrams and unigrams 
# the vocab is all the sentence tokens in the corpus 

lm.fit(train_data, padded_sents)
len(lm.vocab)


# In[44]:


# inspect the model's vocabulary. 
# be sure that a sentence you know exists (from tokenized_text) is in the 
print(lm.vocab.lookup(tokenized_text[0]))


# In[45]:


# see what happens when we include a word that is not in the vocab. 
print(lm.vocab.lookup('then wear the gold hat iphone .'.split()))


# In[46]:


# how many times does alice appear in the model?
print(lm.counts['alice'])

# what is the probability of alice appearing? 
# this is technically the relative frequency of alice appearing 
lm.score('alice')


# In[47]:


# what is the score of 'UNK'? 

lm.score("<UNK>")


# In[48]:


# generate a 20 word sentence starting with the word, 'alice'

print(lm.generate(20, text_seed= 'alice', random_seed=42))


# In[49]:


from nltk.tokenize.treebank import TreebankWordDetokenizer

detokenize = TreebankWordDetokenizer().detokenize

def generate_sent(lm, num_words, text_seed, random_seed=42):
    """
    :param model: An ngram language model from `nltk.lm.model`.
    :param num_words: Max no. of words to generate.
    :param random_seed: Seed value for random.
    """
    content = []
    for token in lm.generate(num_words, text_seed=text_seed, random_seed=random_seed):
        if token == '<s>':
            continue
        if token == '</s>':
            break
        content.append(token)
    return detokenize(content)


# In[50]:


# Now generate sentences that look much nicer. 
generate_sent(lm, 20, text_seed='alice', random_seed = 42)


# In[51]:


# Now generate sentences that look much nicer. 
generate_sent(lm, 40, text_seed='alice', random_seed = 62)


# In[52]:


# Now generate sentences that look much nicer. 
generate_sent(lm, 60, text_seed='alice', random_seed = 82)


# In[ ]:




