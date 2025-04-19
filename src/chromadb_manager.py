from src.config import settings
from src.paths import CHROMA_DB_PATH
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


api_key = settings['openai']
model = 'text-embedding-3-large'
llm_embedding = OpenAIEmbeddings(api_key=api_key, model=model)


class ChromaDBManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=model, api_key=api_key)
        self.vectorstore = Chroma(
            collection_name='test',
            embedding_function=self.embeddings,
            persist_directory=str(CHROMA_DB_PATH),
        )

    def store(
        self,
        texts: list[str],
        ids: list[str],
        metadatas: list[dict],
    ):
        self.vectorstore.add_texts(
            texts=texts,
            ids=ids,
            metadatas=metadatas,
        )

    def find(
        self,
        metadata: dict,
    ):
        result = self.vectorstore.get(
            where=metadata,
            include=["documents"]
        )
        return result["documents"]

    def query(
        self,
        query: str,
        metadata: dict,
        k: int = 2,
    ):
        result = self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=metadata,
        )
        return result

    def drop(
        self,
        metadata: dict,
    ):
        self.vectorstore.delete(
            where=metadata,
        )