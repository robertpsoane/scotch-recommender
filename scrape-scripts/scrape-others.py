from scraping import getAndSave
from merge import mergeSets

config = [
    [
        "indian",
        "https://www.masterofmalt.com/country/indian-whisky/",
        4
    ],
    [
        "welsh",
        "https://www.masterofmalt.com/country/welsh-whisky/",
        2
    ],
    [
        "english",
        "https://www.masterofmalt.com/country/english-whisky/",
        12
    ],
    [
        "canadian",
        "https://www.masterofmalt.com/country/canadian-whisky/",
        8
    ],
    [
        "swedish",
        "https://www.masterofmalt.com/country/swedish-whisky/",
        6
    ],
    [
        "dutch",
        "https://www.masterofmalt.com/country/dutch-whisky/",
        2
    ],
    [
        "southafrican",
        "https://www.masterofmalt.com/country/south-african-whisky/",
        2
    ],
    [
        "australian",
        "https://www.masterofmalt.com/country/australian-whisky/",
        4
    ],
    [
        "kiwi",
        "https://www.masterofmalt.com/country/kiwi-whisky/",
        3
    ]
]
csvs = []
for t in config:
    label = t[0]
    url = t[1]
    np = t[2]
    csvs.append(label+".csv")
    getAndSave(label, url, np)

mergeSets(csvs, "others.csv")
