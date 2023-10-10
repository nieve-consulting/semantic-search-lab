import mlflow


class ProductSearchWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import pandas as pd
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores import Chroma

        # retrieve embedding model
        embedding_model = HuggingFaceEmbeddings(
            model_name=context.artifacts["embedding_model"]
        )

        # retrieve vectordb contents
        self._vectordb = Chroma(
            persist_directory=context.artifacts["chromadb"],
            embedding_function=embedding_model,
        )

        # set number of results to return
        self._max_results = 5

    # define steps to generate results
    # note: query_df expects only one query
    def predict(self, context, query_df):
        # import required libraries
        import pandas as pd

        # perform search on embeddings
        raw_results = self._vectordb.similarity_search_with_score(
            query_df["query"].values[0],  # only expecting one value at a time
            k=self._max_results,
        )

        # get list of scores, descriptions and ids from raw results
        scores, descriptions, names, ids = zip(
            *[
                (
                    r[1],
                    r[0].page_content,
                    r[0].metadata["product_name"],
                    r[0].metadata["product_id"],
                )
                for r in raw_results
            ]
        )

        # reorganized results as a pandas df, sorted on score
        results_pd = pd.DataFrame(
            {
                "product_id": ids,
                "product_name": names,
                "product_description": descriptions,
                "score": scores,
            }
        ).sort_values(axis=0, by="score", ascending=True)

        # set return value
        return results_pd
