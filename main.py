from flask import Flask,render_template
import numpy as np
import GetOldTweets3 as got
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer



app = Flask(__name__)
def analyze_polarity(text):
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(text)
    if ss["compound"] == 0.0:
        sent = "Neutral"
    elif ss["compound"] > 0.0:
        sent = "Postive"
    else:
        sent = "Negative"
    return sent

def TweetExtract(username,maxCount):
    tweetCriteria = got.manager.TweetCriteria().setUsername(username).setTopTweets(True).setMaxTweets(maxCount)
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    tw=[]
    for tweet in tweets:
        tw.append([tweet.text,tweet.date,tweet.favorites,tweet.retweets,tweet.hashtags])
    return tw

@app.route('/display/hourwisesentiment/<string:userData>')
def display(userData):
    return render_template('plot.html', url='/static/images/{}.png'.format(userData))

@app.route('/plot/hourwisesentiment/<string:userData>')
def plot(userData):
    userDataFile = "{}.csv".format(userData)
    df = pd.read_csv(userDataFile)
    df['Polarity'] = np.array([analyze_polarity(text) for text in df['Tweet']])
    stackedAnalysis = pd.crosstab(df["Hour"],df["Polarity"]).apply(lambda x:100*(x/sum(x)),axis=1)
    stackedAnalysis.plot(kind = "bar",figsize = (20,10),color = ['#253494','#41b6c4','#c7e9b4'],linewidth = 1,edgecolor = "#FFFFFF",stacked = True)
    plt.legend(loc = 1,title='Sentiment',title_fontsize=14,prop = {"size" : 12})
    plt.title(r"$\bf{"+'Sentiment Activity By Hour'+"}$"+"\n",fontsize=16)
    plt.ylabel("Sentiment Distribution: Tweet Activity By Hour")
    plt.xlabel("Hour of the Day",fontsize=16)
    plt.savefig('static/images/{}.png'.format(userData))

    return "Plot made"

if __name__ == '__main__':
   app.run(debug=True)