# RAG System Practice

## Attribution

This project uses a subset (10,000 rows) of the `wikipedia-2017-bm25` dataset (https://huggingface.co/datasets/Comet/wikipedia-2017-bm25), created by Comet ML and Vincent Koc, based on Wikipedia 2017 content. The dataset is licensed under [Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0)](https://creativecommons.org/licenses/by-sa/3.0/).

## About This Project

This repository is my personal practice implementation of a Retrieval-Augmented Generation (RAG) system

### How it works

1. Wikipedia titles are embedded using Qwen3-Embedding and stored in Chromadb
2. User query is embedded and used to semantically retrieve titles from Chromadb
3. Retrieved titles are matched to their text
4. Titles and theirs text are reranked for most relevant content
5. Reranked context is fed to the main llm for response generation

Technologies used : Chromadb, qwen3-embedding, qwen3-reranker, qwen3

## License

This project and any modifications are shared under the same [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) license as the original dataset.
