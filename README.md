# Semantic Search Lab – Product Catalog
The idea behind this repo is to leverage an off-the-shelf (pre-trained) language model, in order to infer the meaning of the textual descriptions of the products, enabling semantic searching of the catalogue.

We provide a `docker-compose` file in order to locally set up all the external dependencies required to replicate our results.

For this use case, we will make use of the [Wayfair Annotation Dataset (WANDS)](). This dataset includes over `40,000` unique products with additional metadata, including textual descriptions.

In terms of the language model, we opt for `all-MiniLM-L12-v2`, a fine-tuned, sentence-embedding version of the [MiniLM language model](https://arxiv.org/abs/2002.10957) – a transformer-based language model developed by Microsoft, intended as an efficient alternative to larger models (e.g., MiniLM – 33M parameters, BERT – 109M parameters).

The implementation takes the following approach:
1. Load, pre-process and store the product catalogue using `Spark`.
2. Download the pre-trained sentence embedding model from [HuggingFace Model Hub](https://huggingface.co/docs/hub/models-the-hub), using `sentence_transformers`.
3. Use `langchain` to load the processed data, compute the embeddings for each product and store the computed embeddings in `Chroma`.
4. Use `MLFlow` to build a deployable model.

### 1. Loading, Pre-Processing and Storage of Catalogue Data
First, we need to preprocess the dataset. In this project, we have chosen `Spark` to load the original dataset, process it, and store the results. Due to the limited scope and dataset sizes of this proof of concept, we could have opted for using `Pandas` as a more user-friendly (offers a higher level, simpler API), in-memory option for our data pipeline, as we are not taking advantage of the distributed nature of `Spark`.

Regarding the storage of the preprocessed data, we use `Parquet` – a columnar storage file format that provides efficient compression and encoding schemes.

### 2. Download Language Model
We use the `sentence-transformers` library to download the `all-MiniLM-L12-v2` language model. We can not directly use Hugging Face's Transformers library because it does not provide support for sentence embeddings out of the box. `sentence-transformers` offers a wrapper around Hugging Face Transformers, enabling the use of sentence-level embedding models.

We then read the downloaded model using the `HuggingFaceEmbeddings` `langchain` wrapper. This allows us to leverage the abstractions provided by `langchain` in regards to document processing and, helper utilities for handling embeddings.

In addition, thanks to the unified interface that `langchain` provides through the `HuggingFaceEmbeddings` wrapper, we can further evaluate alternative pre-trained language models preserving the rest of the implementation.

### 3. Compute the Product Embeddings
Using the `langchain` wrappers for document loading and `Chroma`, we load the pre-processed data, compute the product embeddings using our language model, and store the resulting embeddings into Chroma.

### 4. Use MLFlow to build a deployable model
[MLflow](https://mlflow.org/) is an open-source platform for managing the machine learning model lifecycle, spanning the experimentation, reproducibility, and deployment phases. 

For our implementation, we have opted to build our model with `mlflow.pyfunc.PythonModel` – a generic Python model that allows the encapsulation of any Python code that produces a prediction from an input. It defines a generic filesystem format for Python models and provides utilities for saving and loading them to and from this format.

When a model is encapsulated in `mlflow.pyfunc.PythonModel`, it enables:
- Model storage to disk, using the MLFlow model format (using `mlflow.pyfunc.save_model`) – a self-contained format for packaging Python models that allows you to save a model and later load it back to produce predictions.
- Serving predictions via REST API using `mlflow.pyfunc.serve`, or as a batch inference job.

### References
* [LLM Product Search Accelerator](https://github.com/databricks-industry-solutions/product-search)
* [privateGPT](https://github.com/imartinez/privateGPT/)
