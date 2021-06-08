"""
Script to scrape all scotch from masterofmalt.com, does take a while!
"""

from .scraping import getAndSave
from .merge import mergeSets

config = [
    [
        "single-malt-scotch",
        "https://www.masterofmalt.com/country-style/scotch/single-malt-whisky/",
        555
    ],
    [
        "blended-malt-scotch",
        "https://www.masterofmalt.com/country-style/scotch/blended-malt-whisky/",
        20
    ],
    [
        "grain-scotch",
        "https://www.masterofmalt.com/country-style/scotch/grain-whisky/",
        30
    ]
    
]

def scrapeScotch():
    csvs = []
    for t in config:
        label = t[0]
        url = t[1]
        np = t[2]
        csvs.append(label+".csv")
        getAndSave(label, url, np)

    mergeSets(csvs, "scotch.csv")



if __name__ == "__main__":
    scrapeScotch()
