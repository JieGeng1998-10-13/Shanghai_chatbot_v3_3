from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import JSONLoader
import json
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_milvus import Milvus
from uuid import uuid4
from langchain_core.documents import Document



# Load the documents
loader = JSONLoader(file_path="local_file.json", jq_schema=".prizes[]", text_content=False)
documents = loader.load()



# Process the documents to decode Unicode sequences and extract text
for doc in documents:
    content_dict = json.loads(doc.page_content)
    name = content_dict.get('项目名称', '')
    num = content_dict.get('项目编号', '')
    time = content_dict.get('时间', '')
    class1 = content_dict.get('交易分类', '')
    class2 = content_dict.get('项目类型', '')
    class3 = content_dict.get('公告类型', '')
    # Update the page_content with the actual Chinese text
    doc.page_content = name
    doc.metadata["content"] ='项目名称:'+ name + ' ' +'项目编号:'+ num + ' ' + '时间:' + time + ' ' +'交易分类:'+ class1  + ' ' + '项目类型:' + class2 + ' ' + '公告类型:' +       class3




# Initialize embeddings and vector store
embeddings = OllamaEmbeddings(model="bge-m3:latest")



URI = "tcp://47.102.103.246:19530"

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
    collection_name="Noun_Collection_bge_v3",
)

# vector_store_saved = Milvus.from_documents(
#     [Document(page_content="foo!")],
#     embeddings,
#     collection_name="langchain_example",
#     connection_args={"uri": URI},
# )

uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=uuids)