from scraping import getAndSave
from merge import mergeSets

config = [
    [
        "american-rum",
        "https://www.masterofmalt.com/country/american-rum/",
        4
    ],
    [
        "banjan-rum",
        "https://www.masterofmalt.com/country/bajan-rum/",
        7
    ],
    [
        "caribbean-rum",
        "https://www.masterofmalt.com/country/caribbean-rum/",
        11
    ],
    [
        "cuban-rum",
        "https://www.masterofmalt.com/country/cuban-rum/",
        4
    ],
    [
        "dominican-rum",
        "https://www.masterofmalt.com/country/dominican-rum/",
        5
    ],
    [
        "english-rum",
        "https://www.masterofmalt.com/country/english-rum/",
        15
    ],
    [
        "french-rum",
        "https://www.masterofmalt.com/country/french-rum/",
        3
    ],
    [
        "grenadan-rum",
        "https://www.masterofmalt.com/country/grenadan-rum/",
        3
    ],
    [
        "guadeloupe-rum",
        "https://www.masterofmalt.com/country/guadeloupe-rum/",
        4
    ],
    [
        "guyanese-rum",
        "https://www.masterofmalt.com/country/guyanese-rum/",
        7
    ],
    [
        "jamaican-rum",
        "https://www.masterofmalt.com/country/jamaican-rum/",
        7
    ],
    [
        "martinican-rum",
        "https://www.masterofmalt.com/country/martinican-rum/",
        6
    ],
    [
        "mauritian-rum",
        "https://www.masterofmalt.com/country/mauritian-rum/",
        5
    ],
    [
        "scotch-rum",
        "https://www.masterofmalt.com/country/scotch-rum/",
        7
    ]

]
csvs = []
for t in config:
    label = t[0]
    url = t[1]
    np = t[2]
    csvs.append(label+".csv")
    getAndSave(label, url, np)

mergeSets(csvs, "rum-all.csv")
