from langchain_ollama import  OllamaEmbeddings
from langchain_chroma import Chroma


# Initialize embeddings and vector store
embeddings = OllamaEmbeddings(model="bge-m3:latest")
vectorstore = Chroma(embedding_function=embeddings, persist_directory="/mnt/workspace/上海市交通系统交易问答框架/名词向量存储/chroma_non2.db")

# Create a retriever and perform a query
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
#docs = retriever.invoke("新建杭州乘务员工程施工总价承包招标公告")
docs = retriever.invoke("阜阳站信什么锁室内设备招标公告这个项目完整名称是什么，我不太记得")

#docs = retriever.invoke("新建合肥至安庆铁路新合肥西站站房及相关工程第二批建管甲供物资")
# Print the retrieved documents
infer_info =[]
for doc in docs:
    #print(doc.page_content)
    #infer_info.append(doc.page_content)
    infer_info.append(doc.metadata)
print(infer_info)