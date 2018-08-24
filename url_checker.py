import requests
import pymongo

mc = pymongo.MongoClient()
scraper_db = mc['scraper']
sites = scraper_db['sites']


def retrieve_site(url: str) -> bytes:
    for site in sites.find():
        if site['url'] == url:
            return site[data]


def scrape_site(url: str) -> bytes:
    '''return content for url, using cached version if available'''
    data = requests.get(url)
    if data:
        return data
    response = requests.get(url)
    data = response.content
    sites.insert_one({'url': url, 'data': data})
    return data


%time data = scrape_site('https://wikipedia.org/')

[site['url'] for site in sites.find()]
