"""
Hashing script - wasn't initially used for scraping however was applied 
later in preprocessing to give each whisky a unique ID. Makes removing
duplicates easier.  To be used in all update methods.
"""

from hashlib import md5
# Creating an ID for each whisky
def hashEl(name, url):
    """
    MD5 hash of Name and URL
    Hash each individually, use max/min functions to ensure hash 2 
    happens in same order irrespective of which get's input first.
    """
    h1 = md5(name.encode()).hexdigest()
    h2 = md5(url.encode()).hexdigest()
    h3 = max(h1, h2) + min(h1, h2)
    h4 = md5(h3.encode()).hexdigest()
    return h3