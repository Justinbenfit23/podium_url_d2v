#NOTE: 
# SCRAPING DISCLAIMER AND OVERVIEW: 
# Today I was asked to go over a side project I did using a simple doc2vec nlp model 
# to be able to find similarity of businesses based on the html text of their websites.
#  
# This appeared to be a viable scraping project because as long as we have urls we can go
# directly to the website of each of these businesses instead of scraping a search engine like 
# Google which is a good way to get your IP address blacklisted. 
# 
# Furthermore, being that the customers we sell to at Podium are generally not super concerned 
# about safeguarding the information presented on their websites, the chances of them employing 
# anti-scraping measures like Cloudflare are unlikely. 
# This fact,in conjunction with the fact that I'm only hitting each website once, 
# falls into the bucket of ethical scraping (in my opinion at least). 
# Others are free to disagree with me there. 

# I will go over the scraper I built, briefly explain what doc2vec is for those unfamiliar, then show the
# model and touch on a few potential use cases that it could be used for. Feel free to slack me after with
# any questions.  


import numpy as np, pandas as pd
import requests
from bs4 import BeautifulSoup
import concurrent.futures
import time

###################
# Get dataframe loaded and organized
FILENAME = './DATASETS/urls_for_scraping copy.csv' #adjust path according to your environment
dataset = pd.read_csv(FILENAME)
# print(dataset.columns)
# dataset = dataset.loc[:10,]

#NOTE: General data prep, remove nan values in website, standardize website format
dataset = dataset.loc[dataset['Lead_Website_Value'].notna()]
dataset['url'] = dataset['Lead_Website_Value']
dataset.drop(columns=['Lead_Website_Value'],inplace=True)
s = dataset['url']
dataset['w_final'] = np.select([(~s.str.startswith('http') & ~s.str.contains('www') & ~s.str.endswith('.com')),
                                (~s.str.startswith('http') & ~s.str.contains('www')), 
                                (s.str.startswith('www'))], 
                                 ['http://www.' + s + '.com','http://www.' + s, 'http://' + s], s)
# print(dataset.head())
dataset = dataset.drop('url', axis=1)
dataset.rename(columns={'w_final':'url'}, inplace=True)
dataset.reset_index(drop=True, inplace=True)

print("\nSize of input file is ", dataset.shape)
##################
# Scraper to get all raw_html

lst = []
#NOTE: Simple function for scraping (scraping is a good candidate for concurrency because there is a lot of waiting
# on the server if run synchronously)
def scraper (url):
    try:
        page = requests.get(url,timeout=5)
        x = BeautifulSoup(page.text, 'html.parser').get_text(separator=' ')
        print(f"{url} scraped successfully")
    except Exception as e:
        print(e)
        x = 'no response'
    return x, url

#NOTE: Concurrent Scraper runs up to 50 workers asynchronously
start = time.perf_counter()                  
with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    s = list(dataset['url'])
    results = executor.map(scraper, s)
    for r, s in results:
        lst.append(r)
        # print(s)

    # for index, result, s in enumerate(results):
        # print(index, result)
        
finish = time.perf_counter()


ser = pd.Series(lst)
dataset = pd.merge(dataset, ser.rename('raw_html_text'), left_index=True, right_index=True)
dataset.drop('Unnamed: 0', axis=1, inplace=True)
print(f'Finished in {round(finish-start, 2)} second(s)')
# dataset.to_csv('podium_location_urls.csv')
dataset.to_csv('./DATASETS/podium_location_urls.csv')



