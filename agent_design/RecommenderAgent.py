
from time import time
from typing import List, TypedDict
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
N_REC = 10
DEF_PARAMS = {
            "Abv" : [0, 100],
            "Price" : [0, None],
            "Size" : [50, 100]
        }

# Defining types for hinting
class TastingNotes(TypedDict, total=False):
    Nose : str
    Palate : str
    Finish : str
    General : str

class WhiskyDict(TypedDict):
    ID : str
    Type : str
    Name : str
    Description : str
    Tasting_Notes : TastingNotes
    Price : float
    Size : float
    ABV : float
    URL : str

class RecommenderParams(TypedDict, total=False):
    n_recommendations : int or None

class CatPreferences(TypedDict, total=False):
    likes : list[str]
    dislikes : list[str]

class RecommenderPreferences(TypedDict, total=False):
    general : CatPreferences
    nose : CatPreferences
    palate : CatPreferences
    finish : CatPreferences

class RecommenderInput(TypedDict, total=False):
    preferences : CatPreferences
    params : RecommenderParams

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
        revcols = ["ID","Rev.General", "Rev.Nose", "Rev.Palate", "Rev.Finish", 
                                                                    "Rev.n"]
        df = pd.DataFrame(lsts, columns=revcols)
        return df
    
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

    def vectorizeTastingNotes(self, tasting_notes : pd.DataFrame):
        print("\nVectorising tasting notes")
        kws = self.loadKWs()

        # Master of Malt models
        mom_model = self.vectorizeNotesMatrix(tasting_notes, kws)

        nose_model = mom_model["nose"]
        palate_model = mom_model["palate"]
        finish_model = mom_model["finish"]
        general_model = mom_model["general"]

        # Adding vector models to database
        self.dfToSQL(nose_model, "nose_model")
        self.dfToSQL(palate_model, "palate_model")
        self.dfToSQL(finish_model, "finish_model")
        self.dfToSQL(general_model, "general_model")

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

    def getWhiskyByID(self, ID : str) -> WhiskyDict:
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

    def stringToEmbedding(self, string : str, kws : list) -> pd.DataFrame:
        lfv = self.lfv(kws)
        df = lfv.fit([string])
        return df

    def getModelledWhisky(self, model_name : str, ID : str) -> np.ndarray:
        model_table = f"{model_name}_model"
        query = f"SELECT * FROM {model_table} WHERE ID = '{ID}';"
        vec_df = pd.read_sql(query, self.alchemy)
        vec_df = np.array(vec_df.drop(columns="ID"))
        return vec_df

    def vectorizePreferenceIDs(self, user_input : RecommenderPreferences, ids_lst : list) -> tuple[dict, set]:
        """
        Takes a set of preferences, and returns models, vectors 
        and models for each preferences given.
        """
        # Defining caching dictionaries        
        vecs = {}
        models = {}
        ideal_vec_cache = {}
        
        # Adding ',s around ids in list
        ids_lst = ['\'' + w_id + '\'' for w_id in ids_lst]

        # For each model queried, get the vectors for all possible ids 
        # and add/subtract to get an 'ideal' vector
        for model in user_input:
            
            # Getting IDs of like and dislike vectors
            like_ids = user_input[model]["likes"]
            dislike_ids = user_input[model]["dislikes"]
            
            # TODO FIX DIS!!!
            sql_query = f"SELECT * FROM {model}_model WHERE ID in ({', '.join(ids_lst)});"
            table = pd.read_sql(sql_query,self.alchemy)
            
            no_ID = table.drop(columns=["ID"])
            
            
            vector = np.zeros_like(np.array(no_ID.loc[0,:]))
            
            for like_id in like_ids:
                if like_id not in ideal_vec_cache:
                    like_vec = self.getModelledWhisky(model, like_id)
                    ideal_vec_cache[like_id] = like_vec
                else:
                    like_vec = ideal_vec_cache[like_id]
                vector = vector + like_vec
            
            for dislike_id in dislike_ids:
                if dislike_id not in ideal_vec_cache:
                    dislike_vec = self.getModelledWhisky(model, dislike_id)
                    ideal_vec_cache[dislike_id] = dislike_vec
                else:
                    dislike_vec = ideal_vec_cache[dislike_id]
                vector = vector + like_vec
                
            # Normalise the ideal vector
            vecs[model] = normVec(vector)
            models[model] = table
            
        

        # Removing entries from models for whiskies used to get recommendations
        for model in user_input:
            models[model] = models[model][[
                w_id not in ideal_vec_cache.keys() for w_id in table.ID.values
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
        sims = abs(np.matmul(M, np.transpose(vec)))
        simdat = pd.DataFrame(sims, columns=[name + "cossim"])
        simdat.index = df["ID"]
        return simdat

    def dreamDramEmbedding(self, dream_dram : TastingNotes):
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
        
        pd.DataFrame()

    def getPossibleIDs(self, uinput : RecommenderInput) -> tuple[pd.DataFrame,list]:
        """
        Takes a dataframe of whiskys, and filters by set of parameters to
        produce reduced set of IDs that satisfy those parameters.
        """
        query = "SELECT ID FROM whiskys WHERE "

        params = uinput["params"]
        # Iterating over possible parameters - if in params dictionary,
        # check if valid.  If valid, then add to query, else add default
        # to query
        conditions = []
        for param in DEF_PARAMS:
            if (param in params):
                choice = params[param]
                if type(choice[0]) is int:
                    conditions.append(f"{param} >= {choice[0]}")
                elif choice[0] is None:
                    pass
                else:
                    conditions.append(f"{param} >= {DEF_PARAMS[param][0]}")
                if type(choice[1]) is int:
                    conditions.append(f"{param} <= {choice[1]}")
                elif choice[0] is None:
                    pass
                else:
                    conditions.append(f"{param} <= {DEF_PARAMS[param][1]}")
            else:
                conditions.append(f"{param} >= {DEF_PARAMS[param][0]}")
                conditions.append(f"{param} <= {DEF_PARAMS[param][1]}")
        query = query + " AND ".join(conditions) + ";"
         
        # Get all ids from query
        ids = pd.read_sql(query, self.alchemy)
        
        # Process ids for two uses.
        list_ids = list(ids["ID"])

        ids.index = ids.ID
        ids = ids.drop(columns="ID")
        ids
        return ids, list_ids

    def recommendFromVectors(self, ideal_vectors : dict, models : dict) -> list[WhiskyDict]:
        # Getting cosine similarities
        sims = []
        for key in ideal_vectors:
            v = ideal_vectors[key]
            model = models[key]
            similarity = self.sim(v, model, key)
            sims.append(similarity)

        sim_id = ids.join(sims).mean(1).to_frame(name="mean_cossim")
        sorted_sim_id = sim_id.sort_values(by="mean_cossim", ascending=False)
        rec_ids = list(sorted_sim_id[:N_REC].index)
        recommendations = [self.getWhiskyByID(w_id) for w_id in rec_ids]
        return recommendations
        
    def recommend(self, user_input : RecommenderInput) -> list[WhiskyDict]:
        """
        Main recommendation function
            - Takes input in the form of a RecommenderInput dictionary, 
            - Returns a list of Tasting Notes
        """
        # Extracting parameters and filtering possible whiskys
        
        ids, ids_lst = self.getPossibleIDs(user_input)

        # Get set of ideal vectors - one for each relevant model
        ideal_vectors, models = self.vectorizePreferenceIDs(
                                        user_input["preferences"],
                                        ids_lst
                                    )
        recommendations = self.recommendFromVectors(ideal_vectors, models)        
        return recommendations




        
        




if __name__ == "__main__":
    from pprint import pprint
    recommender = WhiskyRecommender()
    # whisky_ids = ["ff6df63a99183d4515cbf7e36a84949c3b761a29cdf7bf13c137e5ce9a91abc5", "e1c6b00e52392df9bbe1ec8227ac8d9b298271b28aad6c4e980c47ff213589dc"]
    # for whisky_id in whisky_ids:
    #     whisky_dict = recommender.getWhiskyByID(whisky_id)
    #     pprint(whisky_dict, sort_dicts=False)
    #     print()

    user_profile = {
        "preferences" : {
            "nose" : {
                "likes" : ["ff6df63a99183d4515cbf7e36a84949c3b761a29cdf7bf13c137e5ce9a91abc5"],
                "dislikes" : []
            },
            "palate" : {
                "likes" : ["ff6df63a99183d4515cbf7e36a84949c3b761a29cdf7bf13c137e5ce9a91abc5"],
                "dislikes" : []
            }
        },
        "params" : {
            "Price" : [0, 30]
        }
    }

    whiskys = recommender.recommend(user_profile)
    for whisky in whiskys:
        pprint(whisky, sort_dicts=False)
        print()
