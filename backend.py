import tweepy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from flask import render_template
from os import environ
import time
from datetime import datetime, timedelta

# import StemmerFactory class
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Machine Learning imports
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

import tensorflow as tf
import string
import re
from string import punctuation
from collections import Counter

#import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# SNA
import networkx as nx
from networkx.readwrite import json_graph
from plotly.offline import download_plotlyjs, plot
import plotly.figure_factory as ff

def main(request):
    model = tf.keras.models.load_model("lstm_model.h5")
    keywords = request.form.get("keywords")
    if keywords is None:
        return render_template("index.html")
    jumlah =request.form.get("tnum")
    if int(jumlah) > 80:
        jumlah = 80
    tanggal = int(request.form['date'])
    
    epochtime = time.localtime()
    jam = time.strftime("%H:%M", epochtime)
    # jam = request.form['time']

    API_Key = environ.get('TW_API_KEY')
    API_Key_Secret = environ.get('TW_API_KEY_SECRET')
    Bearer_Token = environ.get('TW_BEARER')
    Access_Token= environ.get('TW_ACCESS_TOKEN')
    Access_Token_Secret= environ.get('TW_ACCESS_TOKEN_SECRET')

    client = tweepy.Client(bearer_token=Bearer_Token,
                        consumer_key=API_Key,
                        consumer_secret=API_Key_Secret,
                        access_token=Access_Token,
                        access_token_secret=Access_Token_Secret,
                        wait_on_rate_limit=True)
    
    keywords = keywords+" lang:id"
    # a = jam.split(':')
    # time = int(a[0]) - 7
    # date = tanggal
    # if time < 0:
    #     time+=24
    #     tan = tanggal.split("-")
    #     kurang = int(tan[2]) - 1
    #     tan[2] = str(kurang)
    #     date = '-'.join(tan)
    # time = str(time)+":"+a[1]
    
    today = datetime.now()    
    n_days_ago = today - timedelta(days=tanggal)
    tgl = n_days_ago.strftime("%Y-%m-%d")
    
    data = client.get_recent_tweets_count(query=keywords,start_time=tgl+"T"+jam+":00Z")
    count=0
    message,retweet_count,retweet,created_at,user_name,user_id,type_re=[],[],[],[],[],[],[]

    for i in data.data:
        try:
            if i['tweet_count'] >= 100:
                batch = 100
                tweets = client.search_recent_tweets(query=keywords, 
                                                    user_auth=True,
                                                    start_time=i['start'],
                                                    end_time=i['end'],
                                                    tweet_fields=['author_id','context_annotations',
                                                                'created_at','in_reply_to_user_id','public_metrics',
                                                                'referenced_tweets'],
                                                    user_fields=['username'], 
                                                    expansions='author_id',
                                                    max_results= batch)
            elif i['tweet_count'] < 10:
                batch = 10
                tweets = client.search_recent_tweets(query=keywords, 
                                                    user_auth=True,
                                                    start_time=i['start'],
                                                    end_time=i['end'],
                                                    tweet_fields=['author_id','context_annotations',
                                                                'created_at','in_reply_to_user_id','public_metrics',
                                                                'referenced_tweets'],
                                                    user_fields=['username'], 
                                                    expansions='author_id',
                                                    max_results= batch)
            else:
                batch = i['tweet_count']
                tweets = client.search_recent_tweets(query=keywords, 
                                                    user_auth=True,
                                                    start_time=i['start'],
                                                    end_time=i['end'],
                                                    tweet_fields=['author_id','context_annotations',
                                                                'created_at','in_reply_to_user_id','public_metrics',
                                                                'referenced_tweets'],
                                                    user_fields=['username'], 
                                                    expansions='author_id',
                                                    max_results=batch)
            # Get users list from the includes object
            users = {u["id"]: u for u in tweets.includes['users']}
            for i in tweets.data:
                if 'referenced_tweets' in i.data.keys():
                    text = i.data['text']
                    message.append(text)
                    retweet_count.append(i.data['public_metrics']['retweet_count'])
                    typer = i.data['referenced_tweets'][0]['type']
                    if typer == 'replied_to':
                        type_re.append(typer)
                        mess = text.split()
                        retweet.append(mess[0].replace('@',''))
                    else:
                        type_re.append(typer)
                        mess = text.split()
                        retweet.append(mess[1].replace('@','').replace(':',''))
                    user = users[i.author_id]
                    user_name.append(user.username)
                    created_at.append(i.data['created_at'])
                else:
                    message.append(i.data['text'])
                    retweet_count.append(i.data['public_metrics']['retweet_count'])
                    type_re.append(np.NaN)
                    retweet.append(np.NaN)
                    user = users[i.author_id]
                    user_name.append(user.username)
                    created_at.append(i.data['created_at'])
            count += len(tweets.data)

            if count > int(jumlah):
                break
        except:
            continue
    df = pd.DataFrame({'author':retweet,
                   'username':user_name,
                   'retweet_count':retweet_count,
                   'tweets':message,
                   'created_at':created_at,
                   'type':type_re})
    df['created_at'] = pd.to_datetime(df['created_at'], format='%Y-%m-%dT%H:%M:%S.000Z', errors='ignore')
    df = df[df['type']!='quoted']
    df = df.drop_duplicates('tweets')
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
    df['tweet_preprocessed'] = df['tweets'].apply(processTweet)
    factory = StopWordRemoverFactory()
    stopwords = factory.get_stop_words()
    # This code disabled due deployment apps only need run once while build
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    listStopwords = set(stopwords.words('indonesian'))
    listStopwords = list(listStopwords)
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()

    def stoptweet(tweet):
        tweet = stopword.remove(tweet)
        filtered = []
        for txt in tweet:
            if txt not in listStopwords:
                filtered.append(txt)
        text = filtered 
        return tweet
    df['tweet_preprocessed'] = df['tweet_preprocessed'].apply(stoptweet)
    # create stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    def stemtweet(tweet):
        tweet = stemmer.stem(tweet)
        return tweet
    df['tweet_preprocessed'] = df['tweet_preprocessed'].apply(stemtweet)
    df = df.drop_duplicates('tweets')
    df = df.dropna(subset=['tweet_preprocessed'])

    # Visualize word cloud

    list_words=''
    for tweet in df['tweet_preprocessed']:
        tweet = tweet.split()
        for word in tweet:
            list_words += ' '+(word)
            
    wordcloud = WordCloud(width = 600, height = 400, background_color = 'black', min_font_size = 10).generate(list_words)
    plt.figure(figsize=(17,14))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    timestamp = time.strftime("%m_%d_%Y__%H%M%S", epochtime)
    fname = 'wordcloud/'+timestamp+'.png'
    plt.savefig('static/'+fname)

    # Make text preprocessed (tokenized) to untokenized with toSentence Function
    X = df['tweet_preprocessed'] 
    max_features = 5000

    # Tokenize text with specific maximum number of words to keep, based on word frequency
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(X.values)
    X = tokenizer.texts_to_sequences(X.values)
    X = pad_sequences(X,maxlen = 41)
    
    model = tf.keras.models.load_model("lstm_model.h5")
    y_pred = model.predict(X)
    pred = []
    for i in y_pred:
        if i[0] == max(i):
            pred.append(0)
        elif i[1] == max(i):
            pred.append(1)
        else:
            pred.append(2)
    
    neg = pred.count(0)
    net = pred.count(1)
    pos = pred.count(2)

    ttotal=pos+neg+net
    #define data


    #create pie chart
    import plotly.graph_objs as go

    values = [pos, net, neg]
    labels = ['Positif', 'Netral', 'Negatif']
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
    html = df1.to_html()
    # text_file = open("static/assets/html/index.html", "w",encoding="utf-8")
    # text_file.write(html)
    # text_file.close()
    
    df2 = df1.dropna()
    net = nx.from_pandas_edgelist(df2, source="author", target="username")
    # Plot it
    G = nx.convert_node_labels_to_integers(net, first_label=0, ordering='default', label_attribute=None)
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
    layout=dict(title= 'Social Network Analysis',  
                font= dict(family='Balto'),
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
    sna = plot(fig,config={"displayModeBar": False},show_link=False, include_plotlyjs=False, output_type='div')
    
    return {
        "total": ttotal,
        "sna" : sna,
        "sa"  : sa,
        "wc"  : fname,
        "snatab": html,
        "visual": {
              "sna" : request.form.get("sna"),
              "tab" : request.form.get("tab"),
              "wc"  : request.form.get("wc"),
              "pie" : request.form.get("pie"),
        }
    }

    # html_string = '''
    #         <html lang="id">
    #         <head>
                
    #             <meta charset="utf-8" />
    #             <meta name="viewport" content="width=device-width, initial-scale=1" />
    #             <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    #             <!-- Bootstrap CSS -->
    #             <link
    #             href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"
    #             rel="stylesheet"
    #             integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3"
    #             crossorigin="anonymous"
    #             />

    #             <title>Sentiment Analysis Result</title>
    #         </head>
    #         <body>
    #             <!-- Judul -->
    #             <h1 class="text-center" style="padding-top: 1rem">Sentiment Analysis Result</h1>
    #             <p class="text-center text-muted">keyword: '''+keywords+''' jumlah tweet: '''+jumlah+''' waktu: '''+tanggal+" "+jam+''' hingga saat ini </h2>
    #             <!-- End Judul -->
    #             <!-- Visual -->
    #             <section class="visual" style="padding-top: 2rem">
    #             <div class="container">
    #                 <div class="row justify-content-center">
    #                 <div class="col-md-6">
    #                     <h2 class="text-center">Pie Chart</h2>
    #                     <div class="card">
    #                     <div class="card-body">
    #                         '''+sa+'''
    #                     </div>
    #                     </div>
    #                 </div>
    #                 <div class="col-md-6">
    #                     <h2 class="text-center">Word Cloud</h2>
    #                     <div class="card">
    #                     <div class="card-body">
    #                         <img src="/static/assets/img/wc.png" alt="wordcloud" width="100%" height="100%"/>
    #                     </div>
    #                     </div>
    #                 </div>
    #                 </div>
    #             </div>
    #             </section>
    #             <!-- End Visual -->
    #             <!-- SNA -->
    #             <section class="SNA" style="padding-top: 3rem">
    #             <div class="container">
    #                 <div class="row justify-content-center">
    #                 <div class="col-md-12">
    #                     <h2 class="text-center">Social Media Analysis</h2>
    #                     <div class="card">
    #                     <div class="card-body">
    #                         '''+sna+'''
    #                     </div>
    #                     </div>
    #                 </div>
    #                 </div>
    #             </div>
    #             </section>
    #             <!-- End SNA -->
    #             <!-- table -->
    #             <section class="table" style="padding-top: 3rem">
    #             <div class="container">
    #                 <div class="row justify-content-center">
    #                 <div class="col-md-12">
    #                     <h2 class="text-center">Full Data Table</h2>
    #                     <div class="card" style="height: 1050px">
    #                     <div class="card-body">
    #                         <embed src="static/assets/html/index.html" type="text/html" width="100%" height="100%" />
    #                     </div>
    #                     </div>
    #                 </div>
    #                 </div>
    #             </div>
    #             </section>
    #             <script
    #             src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"
    #             integrity="sha384-ka7Sk0Gln4gmtz2MlQnikT1wXgYsOg+OMhuP+IlRH9sENBO0LRn5q+8nbTov4+1p"
    #             crossorigin="anonymous"
    #             ></script>
    #         </body>
    #         </html>
    #         '''
    # print(html_string)
    # with open("templates/out.html", 'w') as f:
    #     f.write(html_string)