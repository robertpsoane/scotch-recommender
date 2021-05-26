from scraping import getAndSave
from merge import mergeSets

config = [
    [
        "brandy",
        "https://www.masterofmalt.com/brandy/",
        70
    ],

]
csvs = []
for t in config:
    label = t[0]
    url = t[1]
    np = t[2]
    csvs.append(label+".csv")
    getAndSave(label, url, np)

#mergeSets(csvs, "gin-all.csv")
