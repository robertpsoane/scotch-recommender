from scraping import getAndSave
from merge import mergeSets

config = [
    [
        "american-white-dog",
        "https://www.masterofmalt.com/country-style/american/white-dog-spirit/",
        3
    ],
    [
        "american-bourbon",
        "https://www.masterofmalt.com/country-style/american/bourbon-whiskey/",
        34
    ],
    [
        "american-single-barrel",
        "https://www.masterofmalt.com/country-style/american/single-barrel-whiskey/",
        2
    ],
    [
        "american-rye",
        "https://www.masterofmalt.com/country-style/american/rye-whiskey/",
        8
    ],
    [
        "american-tenessee",
        "https://www.masterofmalt.com/country-style/american/tennessee-whiskey/",
        5
    ],
    [
        "american-wheat",
        "https://www.masterofmalt.com/country-style/american/wheat-whiskey/",
        2
    ],
    [
        "american-small-batch",
        "https://www.masterofmalt.com/country-style/american/small-batch-whiskey/",
        3
    ],
    [
        "american-corn",
        "https://www.masterofmalt.com/country-style/american/corn-whiskey/",
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

mergeSets(csvs, "american-all.csv")
