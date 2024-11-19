from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import JSONLoader
import json
from langchain_chroma import Chroma
from uuid import uuid4

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
    #doc.page_content = name + ' ' + num + ' ' + time + ' ' + class1  + ' ' + class2 + ' ' + class3
    doc.page_content = name
    doc.metadata["content"] ='项目名称:'+ name + ' ' +'项目编号:'+ num + ' ' + '时间:' + time + ' ' +'交易分类:'+ class1  + ' ' + '项目类型:' + class2 + ' ' + '公告类型:' +       class3



# Initialize embeddings and vector store
embeddings = OllamaEmbeddings(model="bge-m3:latest")

vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory="./chroma_non2.db")




# Create a retriever and perform a query
retriever = vectorstore.as_retriever()
docs = retriever.invoke("新建杭州机辆段机车乘务员待乘楼工程施工总价承包招标公告")

# Print the retrieved documents
for doc in docs:
    print(doc.metadata["content"])
