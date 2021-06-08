"""
Types - used for type hinting
"""

from typing import List, TypedDict

# Defining types for hinting
class TastingNotes(TypedDict, total=False):
    Nose : str
    Palate : str
    Finish : str


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
    Abv : List[int]
    Price : List[int]
    Size : List [int]


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
