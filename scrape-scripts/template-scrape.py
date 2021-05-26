# Template for web scraping and saving:

from scraping import getAndSave

# Suppose we wish to get and scrape all of japanese blended malt, and 
# the url was 
# https://www.masterofmalt.com/country-style/japanese/blended-whisky/

label = "japan-blended-malt"
url = "https://www.masterofmalt.com/country-style/japanese/blended-whisky/"
getAndSave(label, url)