from src.config import settings
from src.paths import CONSULTORIO_PDF_PATH
from src.chromadb_manager import ChromaDBManager

from uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_openai import ChatOpenAI

api_key = settings['openai']
model = 'gpt-4o-mini'
llm = ChatOpenAI(api_key=api_key, model=model)

chromadb_manager = ChromaDBManager()

loader = PyPDFLoader(str(CONSULTORIO_PDF_PATH))

content = loader.load()

text = ""
for page in content:
    text += page.page_content + "\n"

# Using TokenTextSplitter
text_splitter = TokenTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    model_name=model
    )

texts = text_splitter.split_text(text)
uuids = [str(uuid4()) for _ in range(len(texts))]
metadatas = [{'filename': str(CONSULTORIO_PDF_PATH)} for _ in range(len(texts))]

chromadb_manager.store(
    texts=texts,
    ids=uuids,
    metadatas=metadatas
)

query = "cual es el precio de las consultas"
response = chromadb_manager.query(
    query=query,
    metadata={'filename': str(CONSULTORIO_PDF_PATH)},
    k=2
)
print(response)
