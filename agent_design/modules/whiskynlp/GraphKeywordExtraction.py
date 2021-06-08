"""
GraphKeywordExtraction - using graph methods to extract keywords
- Based on RAKE
"""
from .WhiskyLemmatizer import WhiskyLemmatizer
import numpy as np
import pandas as pd
from operator import itemgetter
import string

class GraphKE:

    def __init__(self):
        self.keywords = []
        self.Lemmatizer = WhiskyLemmatizer()
        self.punct = string.punctuation+'’'

    def makeCorpus(self, lst):
        """
        Joins list on a space to form corpus of text
        """
        corp = ' '.join(lst)
        return corp

    def makeCorpusList(self, raw_df, col):
        """
        Extracts column as a list, stripping out punctuation
        """
        df = pd.DataFrame(raw_df[col]).dropna().reset_index()
        out = []
        for row in range(len(df.index)-1):
            # Extracting cell
            
            row_str = df[col][row].lower()
            # Removing punctuation
            row_str = row_str.translate(str.maketrans(' ',' ',self.punct))
            out.append(row_str)
        return out

    def makeNodes(self, corpus):
        """
        Takes a text corpus and using the lemmatizer filters to a set 
        of nodes.  No edges so far - edges to be added when parsing list
        """
        filtered = self.Lemmatizer.tokenFilter(corpus)
        return list(set(filtered))

    def incrementEdge(self, edges, from_idx, to_idx):
        """
        Increments edges in edges dictionary
        """
        if from_idx in edges:
            if to_idx in edges[from_idx]:
                edges[from_idx][to_idx] += 1
            else:
                edges[from_idx][to_idx] = 1
        else:
            edges[from_idx] = {to_idx:1}
        return

    def addElEdges(self, el, nodes, edges, adj, verbose):
        """
        Parses document, adds edges based on co-occurences to edges
        and adj matrix
        """
        filter_el = self.Lemmatizer.tokenFilter(el)
        cands = [
            (token, nodes.index(token)) 
            for token in filter_el if token in nodes
            ]

        n_cands = len(cands)

        for el1 in range(n_cands - 1):
            for el2 in range(el1+1, n_cands):
                node1 = cands[el1][1]
                node2 = cands[el2][1]
                if node1 != node2:
                    from_idx = min(node1, node2)
                    to_idx = max(node1, node2)
                    
                    # Increment edge in adjacency matrix 
                    # (Reflecting undicrectionality of graph)
                    adj[from_idx][to_idx] += 1
                    adj[to_idx][from_idx] += 1

                    self.incrementEdge(edges, from_idx, to_idx)
        return


    def makeEdges(self, lst, nodes, verbose):
        """
        Make edges from a set of nodes and set of documents
        """

        n_nodes = len(nodes)

        # Edges and adjacency matrix to add to
        edges = {}
        adj = np.zeros((n_nodes, n_nodes))

        for el in lst:
            self.addElEdges(el, nodes, edges, adj, verbose)

        # Convert edges from dictionary to list
        edges_list = []
        for start in edges.keys():
            for end in edges[start].keys():
                edge = {
                    "from": start,
                    "to": end,
                    "weight": edges[start][end]
                }
                if verbose:
                    edge["english"] = {
                        "from": nodes[start],
                        "to": nodes[end]
                    }
                edges_list.append(edge)

        return edges_list, adj


    def makeGraph(self, corpus_list, verbose_edges=False, verbose_logging=False):
        """
        MakeGraph - makes co-occurence graph from corpus list 
        """
        corpus = self.makeCorpus(corpus_list)
        nodes = self.makeNodes(corpus)
        if verbose_logging:
            print("Candidate Keywords Selected")
        edges, adj = self.makeEdges(corpus_list, nodes, verbose_edges)
        if verbose_logging:
            print("Edges Created")
        graph = {
            "nodes": nodes,
            "edges": edges,
            "adjacency": adj
        }
        return graph

    def eigenCentralityRank(self, G):
        nodes = G["nodes"]
        adj = G["adjacency"]

        _, evec = np.linalg.eigh(adj)
        abs_ranks = np.abs(evec[:,-1])

        ranked_nodes = list(
            zip(
                nodes, list(abs_ranks)
            )
        )
        ranked_nodes.sort(key=itemgetter(1), reverse=True)
        return ranked_nodes


    def keywordExtract(self, df, col, n_kw=None, verbose_logging=True):
        """
        Extracts n_kw keywords from column of dataframe using co-occurence
        graph methods. 
        """
        if verbose_logging:
            print("Building Corpus")
        self.corpus_list = self.makeCorpusList(df, col)
        if verbose_logging:
            print("Building Graph")
        G = self.makeGraph(self.corpus_list,False,verbose_logging)
        self.G = G
        if verbose_logging:
            print("Ranking Nodes")
        self.ranked_nodes = self.eigenCentralityRank(G)

        if n_kw is not None:
            keywords = self.ranked_nodes[:n_kw]
        else:
            keywords = self.ranked_nodes

        return [word[0] for word in keywords]


    def __repr__(self):
        return "<GraphKeywordExtractor>"
