import pandas as pd
import numpy as np
import tweepy
import matplotlib.pyplot as plt
import sys

from os import environ
import time

# import StemmerFactory class
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Machine Learning imports
import nltk
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

import string
import re
from string import punctuation
from collections import Counter

# SNA
import networkx as nx
from networkx.readwrite import json_graph
from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot, plot
import plotly.figure_factory as ff
init_notebook_mode(connected=True)

def main(request):
  keywords = request.form["keywords"]
  tnum =request.form["tnum"]
  if int(tnum) > 1000:
      tnum = 1000

  # Call twitter api keys
  consumer_key = environ.get('TW_CONSUMER_KEY')
  consumer_secret = environ.get('TW_CONSUMER_SECRET')
  access_token = environ.get('TW_ACCESS_TOKEN')
  access_secret = environ.get('TW_ACCESS_SECRET')
  auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
  auth.set_access_token(access_token, access_secret)
  api = tweepy.API(auth)
  # Search keywords
  tweets = tweepy.Cursor(api.search_tweets,q=keywords,tweet_mode="extended",lang="id").items(int(tnum))
  message,retweet_count,retweet,created_at,user_name,user_id=[],[],[],[],[],[]
  count = 0
  for tweet in tweets:
      count=count+1
      if hasattr(tweet, 'retweeted_status'):
          message.append(tweet.retweeted_status.full_text)
          retweet_count.append(tweet.retweet_count)
          retweet.append(tweet.retweeted_status.user.screen_name)
          created_at.append(tweet.created_at)
          user_name.append(tweet.user.screen_name)
          user_id.append(tweet.user.id)
      else:
          message.append(tweet.full_text)
          retweet_count.append(tweet.retweet_count)
          retweet.append(print(''))
          created_at.append(tweet.created_at)
          user_name.append(tweet.user.screen_name)
          user_id.append(tweet.user.id)
  # insert tweets to database
  for i in range(count):
      data=[message[i], retweet_count[i], retweet[i], created_at[i], user_name[i], user_id[i]]
  # make dataframe
  df=pd.DataFrame({
      'author':retweet,
      'username':user_name,
      'retweet_count':retweet_count,
      'tweets':message,
      'created_at':created_at
  })
  #df = df.sort_values(['created_at'], ascending=[0])
  df1 = df.copy()
  # helper function to clean tweets
  def processTweet(tweet):
      # Remove HTML special entities (e.g. &amp;)
      tweet = re.sub(r'\&\w*;', '', tweet)
      #Convert @username to AT_USER
      tweet = re.sub('@[^\s]+','',tweet)
      # Remove tickers
      tweet = re.sub(r'\$\w*', '', tweet)
      # To lowercase
      tweet = tweet.lower()
      # Remove hyperlinks
      tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
      # Remove hashtags
      tweet = re.sub(r'#\w*', '', tweet)
      # Remove Punctuation and split 's, 't, 've with a space for filter
      tweet = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', tweet)
      # Remove words with 2 or fewer letters
      tweet = re.sub(r'\b\w{1,2}\b', '', tweet)
      # Remove whitespace (including new line characters)
      tweet = re.sub(r'\s\s+', ' ', tweet)
      # Remove single space remaining at the front of the tweet.
      tweet = tweet.lstrip(' ') 
      # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
      tweet = ''.join(c for c in tweet if c <= '\uFFFF')
      return tweet
  # clean dataframe's text column
  df['tweets'] = df['tweets'].apply(processTweet)
  # drop duplicates
  df = df.drop_duplicates('tweets')
  factory = StopWordRemoverFactory()
  stopword = factory.create_stop_word_remover()
  
  def stoptweet(tweet):
    tweet = stopword.remove(tweet)
    replace_list = ['wow semua']
    tweet = re.sub(r'|'.join(map(re.escape, replace_list)), '', tweet)
    return tweet
  df['tweets'] = df['tweets'].apply(stoptweet)
  # create stemmer
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()
  def stemtweet(tweet):
    tweet = stemmer.stem(tweet)
    return tweet
  # clean dataframe's text column
  df['tweets'] = df['tweets'].apply(stemtweet)
  lst = ['']
  lst = [x.strip() for x in lst] 
  # tokenize helper function
  def text_process(raw_text):
      # Check characters to see if they are in punctuation
      nopunc = [char for char in list(raw_text) if char not in string.punctuation]

      # Join the characters again to form the string.
      nopunc = ''.join(nopunc)
      
      # Now just remove any stopwords
      return [word for word in nopunc.lower().split() if word.lower() not in lst]

  # -------------------------------------------

  # tokenize message column and create a column for tokens
  df['tokens'] = df['tweets'].apply(text_process) # tokenize style 1
  df = df[['tweets','tokens']]
  all_words = []
  for line in df['tokens']: 
      all_words.extend(line)  
  # create a word frequency dictionary
  wordfreq = Counter(all_words)
  wordfreq.most_common(10)
  from wordcloud import WordCloud
  wordcloud = WordCloud(width=900,
                        height=500,
                        max_words=500,
                        max_font_size=100,
                        relative_scaling=0.5,
                        colormap='gist_rainbow',
                        normalize_plurals=True).generate_from_frequencies(wordfreq)
  plt.figure(figsize=(17,14))
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")
  
  epochtime = time.localtime()
  timestamp = time.strftime("%m_%d_%Y__%H%M%S", epochtime)
  fname = 'wordcloud/'+timestamp+'.png'
  plt.savefig('static/'+fname)
  model_NB = joblib.load("twitter_sentiment.pkl")
  # run predictions on twitter data
  tweet_preds = model_NB.predict(df['tweets'])

  # append predictions to dataframe
  df_tweet_preds = df.copy()
  df_tweet_preds['predictions'] = tweet_preds
  pos = df_tweet_preds.predictions.value_counts()[0]
  neg = df_tweet_preds.predictions.value_counts()[1]
  

  ttotal=pos+neg
  print('Model predictions: Positives - {}, Negatives - {}'.format(neg,pos))

  import plotly.graph_objs as go

  labels = ['Positif','Negatif']
  values = [int(pos),int(neg)]
  posneg = {'data' : [{'type' : 'pie', 
                       'name' : "Students by level of study",  
                       'labels' : labels,
                       'values' : values,
                       'direction' : 'clockwise',
                       'marker' : {'colors' : ["rgb(251,57,88)", "rgb(0,64,255)"]}}],
                      'layout' : {'title' : ''}}
  sa = plot(posneg,config={"displayModeBar": False}, 
                 show_link=False, 
                 include_plotlyjs=False, 
                 output_type='div')
  # Remove null account
  df1=df1.dropna()
  # Netwrokx
  net = nx.from_pandas_edgelist(df1, source="author", target="username")
  # Plot it
  G = nx.convert_node_labels_to_integers(net, first_label=0, ordering='default', label_attribute=None)
  print(G, file=sys.stderr)
  pos=nx.fruchterman_reingold_layout(G)
  #create labels
  poslabs=nx.fruchterman_reingold_layout(net)
  labels=list(poslabs) + list(' : ')
  #create edges
  Xe=[]
  Ye=[]
  for e in G.edges():
      Xe.extend([pos[e[0]][0], pos[e[1]][0], None])
      Ye.extend([pos[e[0]][1], pos[e[1]][1], None])
  trace_edges=dict(type='scatter',
                  mode='lines',
                  x=Xe,
                  y=Ye,
                  line=dict(width=1, color='rgb(25,25,25)'),
                  hoverinfo='none' 
                  )

  #create nodes
  Xn=[pos[k][0] for k in range(len(pos))]
  Yn=[pos[k][1] for k in range(len(pos))]
  trace_nodes=dict(type='scatter',
                  x=Xn, 
                  y=Yn,
                  mode='markers',
                  marker=dict(showscale=True,size=5,color=[],colorscale='Rainbow',reversescale=True,colorbar=dict(
                      thickness=15,
                      title='Node Connections',
                      xanchor='left',
                      titleside='right')),
                  text=labels,
                  hoverinfo='text')

  #scale color by size
  for node, adjacencies in enumerate(G.adjacency()):
      trace_nodes['marker']['color']+=tuple([len(adjacencies[1])])
  #plot
  axis=dict(showline=False, # hide axis line, grid, ticklabels and  title
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title='' 
            )
  layout=dict(font= dict(family='Balto'),
              width=600,
              height=600,
              autosize=True,
              showlegend=False,
              xaxis=axis,
              yaxis=axis,
              margin=dict(l=40,r=40,b=85,t=100,pad=0,
              ),
              hovermode='closest',
  #     plot_bgcolor='#000000',           
      )
  fig = dict(data=[trace_edges,trace_nodes], layout=layout)
  #run plot
  sna = plot(fig,config={"displayModeBar": False},show_link=False,
            include_plotlyjs=False, output_type='div')
  snatable = ff.create_table(df1)
  snatab = plot(snatable,config={"displayModeBar": False}, 
                show_link=False, 
                include_plotlyjs=False, 
                output_type='div')
          # Save plot to html
  return {
        "total": ttotal,
        "sna" : sna,
        "sa"  : sa,
        "wc"  : fname,
        "snatab": snatab,
        "visual": {
              "sna" : request.form.get("sna"),
              "tab" : request.form.get("tab"),
              "wc"  : request.form.get("wc"),
              "pie" : request.form.get("pie"),
        }
    }