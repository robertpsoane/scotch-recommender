from bs4 import BeautifulSoup
from requests import get
import pandas as pd
import numpy as np
from pprint import pprint 

cols = ["Type","Name","Description","Nose","Palate","Finish", "Price", "Size", "Abv", "URL"]

def getWhiskysFromResultsURL(results_url, cols, wtype):
    pagedf = pd.DataFrame(columns=cols)
    print(f"Getting whisky's from {results_url}.")
    print("Getting the following attributes:")
    pprint(cols)
    try:
        # Get results page
        response = get(results_url)
        page = BeautifulSoup(response.text, 'html.parser')

        # Find whiskys
        whiskys = page.find_all('div', class_="boxBgr product-box-wide h-gutter js-product-box-wide")
        if len(whiskys) > 0:
            # We have found whiskys in results
            for whisky in whiskys:
                url = whisky.a["href"]
                whisky = getWhiskyFromURL(url, cols, wtype)
                pagedf = pagedf.append(whisky)
    except:
        pass
    return pagedf
    
def getWhiskyFromURL(url, cols, wtype):
    try:
        # Get text from page
        response = get(url)
        page = BeautifulSoup(response.text, 'html.parser')
        name = getName(page)
        print(f"- Getting details of {name}.")
        description = getDescription(page)
        nose, palate, finish = getTastingNotes(page)
        price = getPrice(page)
        size, abv = getSizeABV(page)
        return makeDF(wtype, name, description, nose, palate, finish, price, size, abv, url, cols)
    except:
        print(f"Failed to get data from {url}")
        return pd.DataFrame(columns=cols)

def getName(page):
    try:
        name = page.find(id="ContentPlaceHolder1_pageH1").text
        return name
    except:
        return ''

def getDescription(page):
    descs = page.find(id="productDesc").find_all('p')
    desc = ''
    for d in descs:
        desc = desc + d.text
    return desc

def getNose(notes):
    try:
        nose = notes.find(id="ContentPlaceHolder1_ctl00_ctl02_TastingNoteBox_ctl00_noseTastingNote").text[6:]
        return nose
    except:
        return ''

def getPalate(notes):
    try:
        palate = notes.find(id="ContentPlaceHolder1_ctl00_ctl02_TastingNoteBox_ctl00_palateTastingNote").text[8:]
        return palate
    except:
        return ''
    
def getFinish(notes):
    try:
        finish = notes.find(id="ContentPlaceHolder1_ctl00_ctl02_TastingNoteBox_ctl00_finishTastingNote").text[8:]
        return finish
    except:
        return ''

def getTastingNotes(page):
    try:
        notes = page.find(id="ContentPlaceHolder1_ctl00_ctl02_TastingNoteBox_ctl00_productTastingNote2")
        nose = getNose(notes)
        palate = getPalate(notes)
        finish = getFinish(notes)
        return nose, palate, finish
    except:
        return '','',''

    
def getPrice(page)   :
    try:
        reduced = page.find('div',class_="previousprice strike-through")
        if reduced:
            price = reduced.text[1:]
        else:
            priceDiv = page.find_all('div',class_="priceDiv")[0]            
            price = float(priceDiv.span.text[1:])
        return price
    except:
        return np.nan
    
def getSizeABV(page):
    try:
        txt = page.find('span', class_="pageH1ClAbv gold").text.split(', ')
        size = float(txt[0][1:-2])
        abv = float(txt[1][:-2])
        return size, abv
    except:
        return np.nan, np.nan
        
    
def makeDF(wtype, name, desc, nose, palate, finish, price, size, abv, url, cols):
    df = pd.DataFrame(
        [[wtype, name, desc, nose, palate, finish, price, size, abv, url]],
        columns=cols
    )
    return df


def getAndSave(label, base_url, mpages):
    wtype = " ".join(label.split("-"))
    page_numbers = np.arange(1, mpages)

    # empty df
    df = pd.DataFrame(columns=cols)

    for num in page_numbers:
        url = base_url + str(num)
        newdf = getWhiskysFromResultsURL(url, cols, wtype)
        df = df.append(newdf)

    df = df.reset_index()
    df = df[cols]
    df.to_csv(label+".csv")
    

if __name__ == "__main__":
    print("Demo Scraping:")
    url = "https://www.masterofmalt.com/country-style/scotch/single-malt-whisky/1"
    wtype = "single malt"

    df = getWhiskysFromResultsURL(url, cols, wtype)

    df.to_csv("Demo.csv")