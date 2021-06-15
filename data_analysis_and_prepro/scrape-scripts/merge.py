import pandas as pd 
from scraping import cols

def mergeSets(csvs,output):
    df = pd.DataFrame(columns=cols)

    for csv in csvs:
        dat = pd.read_csv(csv)
        df = df.append(dat)
    df = df.reset_index()
    df = df[cols]
    df.to_csv(output)


csvs = [
    "whisky-all.csv", "rum-all.csv", "gin.csv", "brandy.csv", "other-spirits-all.csv"
]
mergeSets(csvs, "all-spirits.csv")
