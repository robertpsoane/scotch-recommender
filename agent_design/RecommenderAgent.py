
from typing import List
import pandas as pd
import numpy as np
import json
import sqlite3
from pandas.io.sql import table_exists

from requests.models import parse_header_links
from whiskynlp.GraphKeywordExtraction import GraphKE
from whiskynlp.Vectorizer import ListFeatureVectorizer
from whiskynlp.WhiskyLemmatizer import WhiskyLemmatizer
import os
from scrapescripts.scrapescotch import scrapeScotch

# Constants:
N_KW = 300
N_GEN = 200


class WhiskyRecommender:

    def __init__(self):
        """
        Init function
        """
        # Adding data paths
        self.moduledir = os.path.dirname(os.path.abspath(__file__))
        self.dbpath = os.path.join(self.moduledir, ".DB", "wdb.db")
        self.kwpath = os.path.join(self.moduledir, ".DB", "kws.json")
        
        # Key external classes
        self.KE = GraphKE()
        self.Lemmatizer = WhiskyLemmatizer()
        self.lfv = ListFeatureVectorizer

        # Check all required data is present.  If it isn't, generate it.
        self.checkInitialSetup()
        
        # Agent is loaded!
        print("Agent loading complete.")

    def __repr__(self):
        return "<WhiskyRecommender>"
    
    def checkInitialSetup(self):
        """
        Checks if signs of initial setup are present. If they aren't, 
        will create database and transfer scotch data to the database.
        """
        # Check existence of database folder
        if not os.path.isdir(".DB"):
            print("First time load. May take longer than usual.")
            os.mkdir(".DB")
        
        # Check existence of sqlite db
        if not os.path.isfile(self.dbpath):
            print("No database on system.")
            spath = os.path.join(self.moduledir, "scotch.csv")
            if os.path.isfile(spath):
                self.makeInitDB(spath)
            else:
                print("No scotch data.  Will scrape from Master of Malt.")
                print("This may take a while...")
                scrapeScotch()
                print("Scraping complete.")
                self.makeInitDB(spath)
        
        if not os.path.isfile(self.kwpath):
            print("No keywords stored. Performing keyword extraction.")
            self.trainModels()
        
        kws = self.loadKWs()
        if ("none-broken" not in kws) or (not kws["none-broken"]):
            print("Keyword JSON broken. Performing keyword extraction.")
            self.trainModels()

        print("Database files available. Agent initialising.")

    def dbcon(self):
        return sqlite3.connect(self.dbpath)

    def dfToSQL(self, df : pd.DataFrame, name : str, exists="replace"):
        """
        Function to add a table to the database, by default replacing 
        the existing table.
        """
        print(f"Adding table {name} to database.")
        con = self.dbcon()
        df.to_sql(name, con, index=False, if_exists=exists)
        con.close()

    def setupReviewTable(self):
        """
        Setup new user review table in db.
        """
        print("Setting up review table.")
        con = self.dbcon()
        cur = con.cursor()
        cur.execute(
            ''' CREATE TABLE reviews(
                ReviewID INTEGER PRIMARY KEY AUTOINCREMENT,
                ProdID STR,
                General VARCHAR(511),
                Nose VARCHAR(511),
                Palate VARCHAR(511),
                Finish VARCHAR(511)
                );
            '''
        )
        con.commit()
        con.close()

    def makeInitDB(self,spath : str):
        """
        Loads from the scotch.csv file into database, 
        Trains models using graph KE
        Saves models to db

        Once this function has been run, should be able to make 
        recommendations.
        """
        print(f"Will setup initial database from {spath}.")
        df = pd.read_csv(spath)
        self.dfToSQL(df, "whiskys")
        self.setupReviewTable()
        self.trainModels()

    def trainModels(self):
        """
        Trains 4 models based on db data:
            - nose
            - palate
            - finish
            - all

        Nose, palate, finish are based on their respective elements of
        tasting notes.  All's keywords are an amalgamation of
        the above, however the BoW model also includes the description
        All three use Reviews data as well.
        """
        tasting_notes = self.getTastingNotes()
        self.performKWExtraction(tasting_notes)
        self.vectorizeTastingNotes(tasting_notes)

    def getTastingNotes(self):
        con = self.dbcon()
        
        tasting_notes = pd.read_sql(
            "SELECT ID, Nose, Palate, Finish, Description FROM whiskys;",
            con
            )
        tasting_notes.fillna(value=np.nan)
        
        con.close()

        return tasting_notes

    def getAllReviews(self):
        """
        Get all reviews as a list of tuples
        """
        con = self.dbcon()
        cur = con.cursor()
        cur.execute(
            """
            SELECT ProdID, General, Nose, Palate, Finish FROM reviews;
            """
        )
        reviews = cur.fetchall()
        con.close()
        return reviews


    def allReviewsAsDict(self):
        """
        Gets all review notes as a concatenated dictionary
        """
        reviews = self.getAllReviews()
        reviews_dict = {}
        for review in reviews:
            if review[0] in reviews_dict:
                review_entry = reviews_dict[review[0]]
                review_entry["general"] = review_entry["general"] \
                                        + " " + review[1]
                review_entry["nose"] = review_entry["nose"] \
                                        + " " + review[2]
                review_entry["palate"] = review_entry["palate"] \
                                        + " " + review[3]
                review_entry["finish"] = review_entry["finish"] \
                                        + " " + review[4]
                review_entry["n"] += 1
            else:
                reviews_dict[review[0]] = {
                    "general": review[1],
                    "nose": review[2],
                    "palate": review[3],
                    "finish": review[4],
                    "n": 1
                }
        return reviews_dict
    
    def performKWExtraction(self, tasting_notes : pd.DataFrame):
        # Perform KW Extraction
        print("\nPerforming Nose Keyword Extraction")
        nose_kws = self.KE.keywordExtract(tasting_notes, "Nose", N_KW)
        print("\nPerforming Palate Keyword Extraction")
        palate_kws = self.KE.keywordExtract(tasting_notes, "Palate", N_KW)
        print("\nPerforming Finish Keyword Extraction")
        finish_kws = self.KE.keywordExtract(tasting_notes, "Finish", N_KW)
        general_kws = list(
            set(nose_kws[:N_GEN] + palate_kws[:N_GEN] + finish_kws[:N_GEN])
            )

        kw_dict = {
            "nose":nose_kws,
            "palate":palate_kws,
            "finish":finish_kws,
            "general":general_kws,
            "none-broken":True
        }

        with open(self.kwpath,"w") as json_out:
            json.dump(kw_dict, json_out)

    def loadKWs(self) ->  dict: 
        with open(self.kwpath) as json_in:
            kws = json.load(json_in)
        return kws

    def readdZeroVectors(self, model : pd.DataFrame, tasting_notes : pd.DataFrame):
        """
        Function to re-introduce missing vectors.  Means that even if 
        there are no notes for a given dram, they will still be query-able
        as part of the suggestion process.  They will have no impact on
        actual output as they have no magnitude.
        """
        # Adding zero vectors
        # # https://stackoverflow.com/a/44318806/14833684
        missing_vals = tasting_notes[
            ~tasting_notes["ID"].isin(model["ID"])].reset_index()
        missing_df = pd.DataFrame(0, index=np.arange(len(missing_vals)), columns=model.columns)
        missing_df["ID"] = missing_vals["ID"]
        model = model.append(missing_df)
        return model
        
    def vectorizeSingleColumn(self, tasting_notes, keywords, name):
        print(f"Vectorising {name} tasting notes")
        vectoriser = self.lfv(keywords)
        df, lst = vectoriser.prepropRemoveNas(tasting_notes, name)
        model = vectoriser.fit(lst)
        model["ID"] = df["ID"]
        model = self.readdZeroVectors(model, tasting_notes)
        return model

    def vectorizeTastingNotes(self,tasting_notes : pd.DataFrame):
        print("\nVectorising tasting notes")
        kws = self.loadKWs()

        # Include review notes!!!!

        tasting_notes["All"] = tasting_notes.Nose + " " +\
             tasting_notes.Palate + " " + tasting_notes.Finish
        nose_model = self.vectorizeSingleColumn(
            tasting_notes, kws["nose"], "Nose")
        palate_model = self.vectorizeSingleColumn(
            tasting_notes, kws["palate"], "Palate")
        finish_model = self.vectorizeSingleColumn(
            tasting_notes, kws["finish"], "Finish")
        general_model = self.vectorizeSingleColumn(
            tasting_notes, kws["general"], "All")

        # Adding vector models to database
        self.dfToSQL(nose_model, "nose_model")
        self.dfToSQL(palate_model, "palate_model")
        self.dfToSQL(finish_model, "finish_model")
        self.dfToSQL(general_model, "general_model")

    def whiskyRow2Dict(self, whisky : list):
        whisky_dict = {
            "ID": whisky[0],
            "Type": whisky[1],
            "Name": whisky[2],
            "Description": whisky[3],
            "Tasting Notes": {
                "Nose": whisky[4],
                "Palate": whisky[5],
                "Finish": whisky[6]
            },
            "Price": whisky[7],
            "Size": whisky[8],
            "ABV": whisky[9],
            "URL": whisky[10]
        }
        return whisky_dict

    def getWhiskyByID(self, ID : str):
        """
        Query database for whisky by ID.
        Format in dictionary
        """
        con = self.dbcon()
        cur = con.cursor()
        cur.execute("SELECT * FROM whiskys WHERE ID=:id", {"id":ID})
        whisky = cur.fetchone()
        con.close()
        try:
            whisky = self.whiskyRow2Dict(whisky)
        except TypeError:
            whisky={}
        return whisky

    def review2DB(self, review : dict) -> None:
        """
        Takes review of the form
        {
            "prod_id": ...
            "general": ...
            "nose": ...
            "palate": ...
            "finish": ...
        }
        and adds to database.
        """
        
        con = self.dbcon()
        cur = con.cursor()
        cur.execute(
            """ 
            INSERT INTO reviews 
                (
                    ProdID,
                    General,
                    Nose,
                    Palate,
                    Finish
                )
            VALUES
                (
                    :prod_id,
                    :general,
                    :nose,
                    :palate,
                    :finish

                );
            """,
            review
        )
        con.commit()
        con.close()

    def stringToEmbedding(self, string : str, kws : list):
        lfv = self.lfv(kws)
        df = lfv.fit([string])
        return df

    def dreamDramEmbedding(self, dream_dram : dict):
        """
        Takes dictionary of the following form:
        {
            "Nose": nose dream dram,
            "Palate": dream palate,
            "Finish": dream finish
        }
        """
        # kws = self.loadKWs()
        # embeddings = {
        #     "Nose":self.stringToEmbedding
        # }
        # return embeddings
        pass





if __name__ == "__main__":
    from pprint import pprint
    recommender = WhiskyRecommender()

    whisky_id = "ff6df63a99183d4515cbf7e36a84949c3b761a29cdf7bf13c137e5ce9a91abc5"
    whisky_dict = recommender.getWhiskyByID(whisky_id)
    pprint(whisky_dict, sort_dicts=False)

    review1 = {
        "prod_id": "ff6df63a99183d4515cbf7e36a84949c3b761a29cdf7bf13c137e5ce9a91abc5",
        "general": "Very nice, heavily peated.",
        "nose": "I don't know",
        "palate": "I don't know",
        "finish": "I don't know"
    }
    review2 = {
        "prod_id": "ff6df63a99183d4515cbf7e36a84949c3b761a29cdf7bf13c137e5ce9a91abc5",
        "general": "Very nice, too peated.",
        "nose": "I don't know",
        "palate": "I don't know",
        "finish": "I don't know"
    }
    recommender.review2DB(review1)
    recommender.review2DB(review2)

    pprint(recommender.getReviewNotes(), sort_dicts=False)
