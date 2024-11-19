from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import JSONLoader
import json
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_milvus import Milvus
from uuid import uuid4
from langchain_core.documents import Document


# Load the documents
loader = JSONLoader(file_path="中标总表.json", jq_schema=".prizes[]", text_content=False)
documents = loader.load()



# Process the documents to decode Unicode sequences and extract text
for doc in documents:
    content_dict = json.loads(doc.page_content)
    name = content_dict.get('中标人名称', '')
    prizes = content_dict.get('价格（万）', '')
    info = content_dict.get('主要标的信息', '')
    web_url = content_dict.get('页面网址', '')
    num = content_dict.get('项目编号', '')
    pro_name = content_dict.get('项目名称', '')
    date = content_dict.get('日期', '')
    pro_class = content_dict.get('公路项目', '')
    comp_url = content_dict.get('爱企查网址', '')
    com_ui = content_dict.get('公司官网', '')
    # Update the page_content with the actual Chinese text
    doc.page_content = name
    doc.metadata["content"] = '中标人名称：' + str(name) + ' ' +'价格（万）：'+ str(prizes) + ' ' +'主要标的信息：'+ str(info) + ' ' +'页面网址：' + str(web_url)  + ' ' +         '项目编号：' + str(num) + ' ' + '项目名称：'+pro_name + ' ' +'日期：'+ str(date) + ' ' + '公路项目：'+ str(pro_class) + ' ' + '爱企查网址'+str(comp_url)  + ' ' +        '公司官网：'+str(com_ui)




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