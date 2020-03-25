import logging
import os
os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/jdk-12.0.1.jdk/Contents/Home"
from pyserini.search import pysearch

class AnseriniSCorer():
    def __init__(self, index_path):
        self.searcher = pysearch. \
            SimpleSearcher(index_path)

    def predict(self, sessions, prediction_cols):
        all_preds = []
        cache = {}
        missed = 0
        for _, r in sessions.iterrows():
            preds_query = []
            query = r["query"].encode('utf-8')
            if query in cache:
                hits = cache[query]
            else:
                hits = self.searcher.search(query, k=1000)
            scores = {}
            for hit in hits:
                scores[hit.content] = hit.score
            for document in [r['relevant_doc']] + \
                            [r[c] for c in prediction_cols]:
                if document in scores:
                    preds_query.append(scores[document])
                else:
                    missed+=1
                    preds_query.append(0)
            all_preds.append(preds_query)
        logging.info("missed docs " +str(missed/(len(all_preds[0])*len(all_preds))))
        return all_preds

class BM25(AnseriniSCorer):

    def __init__(self, *args):
        super().__init__(*args)

    def fit(self, _):
        self.searcher.set_bm25_similarity(0.9, 0.4)

class RM3(AnseriniSCorer):

    def __init__(self, *args):
        super().__init__(*args)

    def fit(self, _):
        self.searcher.set_bm25_similarity(0.9, 0.4)
        self.searcher.set_rm3_reranker(10, 10, 0.5)

class QL(AnseriniSCorer):

    def __init__(self, *args):
        super().__init__(*args)

    def fit(self, _):
        self.searcher.set_lm_dirichlet_similarity(1000.0)
