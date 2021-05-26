from scraping import getAndSave
from merge import mergeSets

config = [
    [
        "cognac",
        "https://www.masterofmalt.com/cognac/",
        36
    ],
    [
        "vodka",
        "https://www.masterofmalt.com/vodka/",
        51
    ],
    [
        "absinth",
        "https://www.masterofmalt.com/absinthe/",
        5
    ],
    [
        "armagnac",
        "https://www.masterofmalt.com/armagnac/",
        21
    ],
    [
        "calvados",
        "https://www.masterofmalt.com/calvados/",
        7
    ],
    [
        "mezcal",
        "https://www.masterofmalt.com/mezcal/",
        13
    ],
    [
        "mead",
        "https://www.masterofmalt.com/mead/",
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

mergeSets(csvs, "other-spirits-all.csv")
