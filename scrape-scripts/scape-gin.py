from scraping import getAndSave
from merge import mergeSets

config = [
    [
        "gin",
        "https://www.masterofmalt.com/gin/",
        154
        
    ]

]
csvs = []
for t in config:
    label = t[0]
    url = t[1]
    np = t[2]
    csvs.append(label+".csv")
    getAndSave(label, url, np)

#mergeSets(csvs, "gin-all.csv")
