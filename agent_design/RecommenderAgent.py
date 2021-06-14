
from time import time
from typing import List, TypedDict
import pandas as pd
import numpy as np
import json
import sqlite3
from pandas.io.sql import table_exists
from requests import models
import random
from requests.models import parse_header_links
from modules.whiskynlp.GraphKeywordExtraction import GraphKE
from modules.whiskynlp.Vectorizer import ListFeatureVectorizer
from modules.whiskynlp.WhiskyLemmatizer import WhiskyLemmatizer
import os
from modules.scraping.scrapescotch import scrapeScotch
from modules.scraping.scraping import getUpdates
from modules.types import *
import warnings
warnings.filterwarnings('ignore')


# Constants:
N_KW = 300
N_GEN = 200
N_REC = 10
DEF_PARAMS = {
            "Abv" : [0, 100],
            "Price" : [0, 100000],
            "Size" : [50, 100]
        }

UPDATE_N = 3

# Constants for managing reviews - if set to True, will include reviews
# in model, else will only use MoM tasting notes
INC_REVIEWS = True
MOM_MIN_WEIGHTING = 50

# Useful vector functions:
def normVec(v):
    norm = np.linalg.norm(v)
    if norm != 0:
        v = v / norm
    return v

# Whisky Recommender 
class WhiskyRecommender:

    def __init__(self):
        """
        Init function
        """
        # Adding data paths
        self.moduledir = os.path.dirname(os.path.abspath(__file__))
        self.dbpath = os.path.join(self.moduledir, ".DB", "wdb.db")
        self.kwpath = os.path.join(self.moduledir, ".DB", "kws.json")
        self.alchemy = "sqlite:///"+self.dbpath
        self.spath = os.path.join(self.moduledir, "scotch.csv")

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
    
    ################################
    ### General useful functions ###
    ################################

    def dbcon(self) -> sqlite3.Connection:
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

    def whiskyRow2Dict(self, whisky : list) -> WhiskyDict:
        whisky_dict = {
            "ID": whisky[0],
            "Type": whisky[1],
            "Name": whisky[2],
            "Description": whisky[3],
            "Tasting_Notes": {
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

    def stringToEmbedding(self, string : str, kws : list) -> pd.DataFrame:
        """
        Converts string to a dataframe based on a bag of words passed in.
        """
        lfv = self.lfv(kws)
        df = lfv.fit([string])
        return df

    ################################
    ######  Setup  functions  ######
    ################################
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
            spath = self.spath
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
    
    ################################
    #####  Training  Functions  ####
    ################################
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
        reviews = self.allReviewsConcatDF()
        self.performKWExtraction(tasting_notes)
        self.vectorizeTastingNotes(tasting_notes, reviews)

    def getTastingNotes(self):
        con = self.dbcon()
        tasting_notes = pd.read_sql(
            "SELECT ID, Nose, Palate, Finish, Description FROM whiskys;",
            con
            )
        tasting_notes.fillna(value=np.nan)
        con.close()
        return tasting_notes

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

    def readdZeroVectors(self, model : pd.DataFrame, tasting_notes : pd.DataFrame) -> pd.DataFrame:
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
        
    def vectorizeSingleColumn(self, tasting_notes : pd.DataFrame,
                            keywords : dict, name : str) -> pd.DataFrame:
        print(f"Vectorising {name} tasting notes")
        vectoriser = self.lfv(keywords)
        df, lst = vectoriser.prepropRemoveNas(tasting_notes, name)
        model = vectoriser.fit(lst)
        model["ID"] = df["ID"]
        model = self.readdZeroVectors(model, tasting_notes)
        return model

    def vectorizeNotesMatrix(self, notes : pd.DataFrame, kws : dict) -> dict:
        notes["All"] = notes.Description + " " \
                            + notes.Nose + " " \
                            + notes.Palate + " " \
                            + notes.Finish
        nose_model = self.vectorizeSingleColumn(
            notes, kws["nose"], "Nose")
        palate_model = self.vectorizeSingleColumn(
            notes, kws["palate"], "Palate")
        finish_model = self.vectorizeSingleColumn(
            notes, kws["finish"], "Finish")
        general_model = self.vectorizeSingleColumn(
            notes, kws["general"], "All")
        model_dict = {
            "nose": nose_model,
            "palate": palate_model,
            "finish": finish_model,
            "general": general_model
        }
        return model_dict

    def sumReviews(self, mom : pd.DataFrame, rev : pd.DataFrame, reviews : pd.DataFrame, name) -> pd.DataFrame:
        """
        sumReviews - sums reviews into MoM vectors, weighted in favour of MoM.
        """
        if INC_REVIEWS:
            print(f"Summing reviews into Master of Malt model for {name}")
            # Getting feature cols
            cols = mom.drop(columns="ID").columns

            # Setting ID as index
            mom.index=mom.ID
            rev.index = rev.ID

            # Selecting number of reviews from reviews dataframe, and finding weighting factor
            n = reviews[["n"]]
            n["n"] = n["n"].apply(lambda n: min(1, n/MOM_MIN_WEIGHTING))
            
            # Adding weighted reviews to mom vectors, and re-normalising
            rev[cols] = rev[cols].add(mom[cols]).dropna().apply(lambda row: normVec(row), axis=1)
            
            # Replacing relevant vectors with amalgamated ones which include reviews
            mom.loc[rev.index] = rev
            # Drop ID and reset index       
            mom = mom.drop(columns="ID").reset_index()
        else:
            pass

        return mom

    def vectorizeTastingNotes(self, tasting_notes : pd.DataFrame, reviews : pd.DataFrame):
        print("\nVectorising tasting notes")
        kws = self.loadKWs()

        # Master of Malt models
        print("MoM Tasting Notes")
        mom_model = self.vectorizeNotesMatrix(tasting_notes, kws)
        print()
        print("Reviews")
        reviews_model = self.vectorizeNotesMatrix(reviews, kws)
        print()
        
        nose_model = self.sumReviews(mom_model["nose"], reviews_model["nose"], reviews, "nose")
        palate_model = self.sumReviews(mom_model["palate"], reviews_model["palate"], reviews, "palate")
        finish_model = self.sumReviews(mom_model["finish"], reviews_model["finish"], reviews, "finish")
        general_model =self.sumReviews(mom_model["general"], reviews_model["general"], reviews, "general")

        # Adding vector models to database
        self.dfToSQL(nose_model, "nose_model")
        self.dfToSQL(palate_model, "palate_model")
        self.dfToSQL(finish_model, "finish_model")
        self.dfToSQL(general_model, "general_model")
    
    ################################
    ## Training/Reviews Functions ##
    ################################
    def getAllReviews(self) -> list:
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

    def allReviewsConcatDict(self) -> dict :
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

    def allReviewsConcatDF(self) -> pd.DataFrame:
        """
        Returns concatenated reviews as dataframe
        """
        reviews = self.allReviewsConcatDict()
        lsts = [
                [
                    k,
                    reviews[k]['general'],
                    reviews[k]['nose'],
                    reviews[k]['palate'],
                    reviews[k]['finish'],
                    reviews[k]['n']
                ]
                for k in reviews
            ]
        revcols = ["ID","Description", "Nose", "Palate", "Finish", "n"]
        df = pd.DataFrame(lsts, columns=revcols)
        return df

    def addReview(self, review : dict) -> None:
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
        required_keys = ["general", "nose", "palate", "finish"]
        for k in required_keys:
            if k not in review:
                review[k] = ""
        con = self.dbcon()
        cur = con.cursor()
        cur.execute(
            """ 
            INSERT INTO reviews 
                (ProdID, General,  Nose, Palate, Finish)
            VALUES
                ( :prod_id, :general, :nose, :palate, :finish );
            """,
            review
        )
        con.commit()
        con.close()
    
    ################################
    ####### Update Functions #######
    ################################

    def updateWhiskies(self):
        """
        Update whiskys from Master of Malt, add to database and retrain
        model.
        Applies following process:
        - Get all IDs currently in database
        - Scrape MoM new whisky pages, for each whisky check if ID in
        database
        - If in db, stop scraping
        - Add each whisky to a new csv - merge with scotch
        - Add each whisky to db
        - Retrain model.
        """
        ids = self.getAllIDs()
        update_df = getUpdates("updates", ids, UPDATE_N)
        self.trainModels()
        scotch = pd.read_csv(self.spath)
        scotch.append(update_df)
        scotch.to_csv(self.spath, index=False)
        self.dfToSQL(update_df, "whiskys", exists='append')
        self.trainModels()
        return

    ################################
    ####  DB  Search  Functions ####
    ################################
    def loadModelledWhiskys(self, model : str, ids : list[str]) -> pd.DataFrame:
        """
        Get all modelled whiskys of given ids in a dataframe.
        Takes the name of a model and a list of ids
        """
        sql_query = f"SELECT * FROM {model}_model WHERE ID in ({', '.join(ids)});"
        return pd.read_sql(sql_query,self.alchemy)

    def getWhiskyByID(self, ID : str) -> WhiskyDict:
        """
        Query database for whisky by ID.
        Format in dictionary
        """
        con = self.dbcon()
        cur = con.cursor()
        cur.execute(
            """
            SELECT  `ID`, `Type`, `Name`, `Description`, `Nose`, `Palate`,
                    `Finish`, `Price`, `Size`, `ABV`, `URL`
            FROM whiskys
            WHERE ID=:id""", 
            {"id":ID}
        )
        whisky = cur.fetchone()
        con.close()
        try:
            whisky = self.whiskyRow2Dict(whisky)
        except TypeError:
            whisky={}
        return whisky
    
    def searchByURL(self, URL : str) -> list[dict]:
        con = self.dbcon()
        cur = con.cursor()
        cur.execute(
            f"""
            SELECT `ID`, `Name`
            FROM `whiskys` 
            WHERE `URL` = :url
            """,
            {
                "url":URL
            }
        )
        whiskys = cur.fetchall()
        con.close()

        try:
            whiskys = list(
                whiskys[0]
            )
        except IndexError:
            whiskys = []
            
        return whiskys

    def searchWhiskys(self, term : str, col : str) -> list[WhiskyDict]:
        con = self.dbcon()
        cur = con.cursor()
        cur.execute(
            f"""
            SELECT  `ID`, `Type`, `Name`, `Description`, `Nose`, `Palate`,
                    `Finish`, `Price`, `Size`, `ABV`, `URL` 
            FROM `whiskys` 
            WHERE {col}
            LIKE :term ;""",
        {
            "term":'%'+term+'%'
        }
        )

        ws = cur.fetchall()
        con.close()
        
        try:
            whiskys = [self.whiskyRow2Dict(w) for w in ws]
        except TypeError:
            whiskys = []

        whiskys = [self.whiskyRow2Dict(w) for w in ws]
        return whiskys

    def searchByDesc(self, term : str) -> List[WhiskyDict]:
        whiskys = self.searchWhiskys(term, "Description")
        return whiskys
    
    def searchByName(self, term : str) -> list[WhiskyDict]:
        whiskys = self.searchWhiskys(term, "Name")
        return whiskys

    def getAllIDs(self) -> list:
        con = self.dbcon()
        cur = con.cursor()
        cur.execute("SELECT ID FROM whiskys")
        ids = cur.fetchall()
        con.close()
        return [w_id[0] for w_id in ids]


    ################################
    # Recommendation Aux Functions #
    ################################
    def vectorizePreferenceIDs(self, user_input : RecommenderPreferences, ids_lst : list) -> tuple[dict, set]:
        """
        Takes a set of preferences, and returns models, vectors 
        and models for each preferences given.
        """
        # Defining caching dictionaries        
        vecs = {}
        models = {}
        ids = set()

        # Getting model ids to a list - ensure they are pulled by the 
        # one mega-query
        # Previously querying each ID, but that was taking a very
        # long time
        for model in user_input:
            for key in user_input[model]:
                for iid in user_input[model][key]:
                    if iid not in ids_lst:
                        ids_lst.append(iid)
        # Adding ',s around ids in list
        ids_lst = ['\'' + w_id + '\'' for w_id in ids_lst]

        # For each model queried, get the vectors for all possible ids 
        # and add/subtract to get an 'ideal' vector
        for model in user_input:
            
            # Getting IDs of like and dislike vectors
            like_ids = user_input[model]["likes"]
            dislike_ids = user_input[model]["dislikes"]
            
            # Get all vectors from model with IDs that we want
            table = self.loadModelledWhiskys(model, ids_lst)
            
            no_ID = table.drop(columns=["ID"])
            
            
            vector = np.zeros_like(np.array(no_ID.loc[0,:]))
            
            for like_id in like_ids:
                like_vec = np.array(no_ID[table["ID"] == like_id])
                vector = vector + like_vec
                ids.add(like_id)
            
            for dislike_id in dislike_ids:
                dislike_vec = np.array(no_ID[table["ID"] == dislike_id])
                vector = vector + dislike_vec
                ids.add(dislike_id)
                
            # Normalise the ideal vector
            vecs[model] = normVec(vector)
            models[model] = table
            
        # Removing entries from models for whiskies used to get recommendations
        for model in user_input:
            models[model] = models[model][[
                w_id not in ids for w_id in models[model].ID.values
                ]]
        
        
        return vecs, models

    def sim(self, vec : np.ndarray, df : pd.DataFrame, name :str = "") -> pd.DataFrame:
        """
        Similarity function.  Calculates cosine similarites between a 
        dataframe of vectors (plus ids - ids get stripped prior to
        calculation) and a vector.

        Note - assumes normalised vectors as we are simply finding the 
        scalar product between each row and the vector.  Would be more 
        computationally intensive to also normalise the output.
        """
        
        M = np.array(df.drop(columns=["ID"]))
        sims = np.matmul(M, np.transpose(vec))
        simdat = pd.DataFrame(sims, columns=[name + "cossim"])
        simdat.index = df["ID"]
        return simdat

    def getPossibleIDs(self, params : RecommenderParams) -> tuple[pd.DataFrame,list]:
        """
        Takes a dataframe of whiskys, and filters by set of parameters to
        produce reduced set of IDs that satisfy those parameters.
        """
        query = "SELECT ID FROM whiskys WHERE "

        # Iterating over possible parameters - if in params dictionary,
        # check if valid.  If valid, then add to query, else add default
        # to query
        conditions = []
        for param in DEF_PARAMS:
            def_choice = DEF_PARAMS[param]
            if (param in params):
                choice = params[param]
                if type(choice[0]) is int:
                    conditions.append(f"{param} >= {choice[0]}")
                elif choice[0] is None:
                    pass
                else:
                    conditions.append(f"{param} >= {def_choice[0]}")
                if type(choice[1]) is int:
                    conditions.append(f"{param} <= {choice[1]}")
                elif choice[1] is None:
                    pass
                else:
                    conditions.append(f"{param} <= {def_choice[1]}")
            else:
                conditions.append(f"{param} >= {def_choice[0]}")
                conditions.append(f"{param} <= {def_choice[1]}")
        query = query + " AND ".join(conditions) + ";"
         
        # Get all ids from query
        ids = pd.read_sql(query, self.alchemy)
        
        # Process ids for two uses.
        list_ids = list(ids["ID"])

        ids.index = ids.ID
        ids = ids.drop(columns="ID")
        ids
        return ids, list_ids

    def recommendFromVectors(self, ideal_vectors : dict, models : dict, ids : pd.DataFrame) -> list[WhiskyDict]:
        # Getting cosine similarities
        sims = []
        for key in ideal_vectors:
            v = ideal_vectors[key]
            model = models[key]
            similarity = self.sim(v, model, key)
            sims.append(similarity)

        sim_id = ids.join(sims).mean(1).to_frame(name="mean_cossim")
        sorted_sim_id = sim_id.sort_values(by="mean_cossim", ascending=False)
        rec_ids = list(sorted_sim_id.index)
        if len(rec_ids) > N_REC:
            rec_ids = rec_ids[:N_REC]
        recommendations = [self.getWhiskyByID(w_id) for w_id in rec_ids]
        return recommendations
    
    ################################
    ### Dream Dram Aux Functions ###
    ################################
    def getDreamDramVector(self, dream_dram : TastingNotes, ids : list) -> tuple[dict, set]:
        

        # Defining caching dictionaries        
        vecs = {}
        models = {}
        ids = ['\'' + w_id + '\'' for w_id in ids]

        kws = self.loadKWs()

        for model in dream_dram:
            # Modelling dream dram as a vector
            lfv = self.lfv(kws[model.lower()])
            vecs[model] = normVec(
                    np.array(
                        lfv.fit(dream_dram[model])
                    )
            )

            # Loading all possible models as a dataframe
            table = self.loadModelledWhiskys(model, ids)
            
            models[model] = table
        
        return vecs, models   

    #################################
    # Recommendation Main Functions #
    #################################
    def recommend(self, user_input : RecommenderInput) -> list[WhiskyDict]:
        """
        Main recommendation function
            - Takes input in the form of a RecommenderInput dictionary, 
            - Returns a list of Tasting Notes
        """
        # Extracting parameters and filtering possible whiskys
        params = user_input["params"]
        user_prefs = user_input["preferences"]

        ids, ids_lst = self.getPossibleIDs(params)
        
    
        # Get set of ideal vectors - one for each relevant model
        ideal_vectors, models = self.vectorizePreferenceIDs(
                                        user_prefs,
                                        ids_lst
                                    )
        
        recommendations = self.recommendFromVectors(ideal_vectors, models, ids)   
        
        
        return recommendations

    def recommendDD(self, user_input : RecommenderInput) -> list[WhiskyDict]:
        """
        Dream Dram recommendation function
        """
        params = user_input["params"]
        user_prefs = user_input["preferences"]

        # Format Dream Dram as dataframe
        user_prefs = {key : [user_prefs[key]] for key in user_prefs}
        
        
        # Getting possible IDs of dream dram
        ids, ids_lst = self.getPossibleIDs(params)

        # Get set of ideal vectors - one for each relevant model
        ideal_vectors, models = self.getDreamDramVector(
                                        user_prefs,
                                        ids_lst
                                    )
        
        recommendations = self.recommendFromVectors(ideal_vectors, models, ids) 
        
        return recommendations



# TODO 
# - Add reviews to models

        


class BaselineRecommender(WhiskyRecommender):
    """
    Baseline Recommender class
    Inherits from whisky recommender, and has access to same databases
    etc, however instead of using NLP to find recommendations, gets a
    random set of N_REC recommendations and returns them in same format

    To be used as a baseline for comparing results.
    """

    def __init__(self):
        WhiskyRecommender.__init__(self)

    def recommend(self, user_input: RecommenderInput) -> list[WhiskyDict]:
        """
        Ranomd recommender - gets ids which satisfy requirements (eg 
        price range etc) and then randomly selects N_REC
        """
        _, ids = self.getPossibleIDs(user_input["params"])
        if len(ids) > N_REC:
            rec_ids = random.sample(ids, N_REC)
        else:
            rec_ids = ids

        recs = [self.getWhiskyByID(w_id) for w_id in rec_ids]
        return recs

    def recommendDD(self, user_input : RecommenderInput) -> list[WhiskyDict]:
        return self.recommend(user_input)


if __name__ == "__main__":
    recommender = WhiskyRecommender()
    print(recommender.updateWhiskies())
        
