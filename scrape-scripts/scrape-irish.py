from scraping import getAndSave
from merge import mergeSets

config = [
    [
        "irish-single-malt",
        "masterofmalt.com/country-style/irish/single-malt-whiskey/",
        9
    ],
    [
        "irish-single-potstill",
        "https://www.masterofmalt.com/country-style/irish/single-pot-still-whiskey/",
        4
    ],
    [
        "irish-blended",
        "https://www.masterofmalt.com/country-style/irish/blended-whiskey/",
        11
    ],
    [
        "irish-grain",
        "https://www.masterofmalt.com/country-style/irish/grain-whiskey/",
        2
    ]

]
csvs = []
for t in config:
    label = t[0]
    url = t[1]
    np = t[2]
    csvs.append(label+".csv")
    getAndSave(label, url, np)

mergeSets(csvs, "irish-all.csv")
