from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_milvus import Milvus

# Initialize embeddings and vector store
embeddings = OllamaEmbeddings(model="bge-m3:latest")
# vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_non2.db")
#
# # Create a retriever and perform a query
# retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
# #docs = retriever.invoke("新建杭州乘务员工程施工总价承包招标公告")
# docs = retriever.invoke("阜阳站信什么锁室内设备招标公告这个项目完整名称是什么，我不太记得")
# # Print the retrieved documents
# infer_info =[]
# for doc in docs:
#     #print(doc.page_content)
#     infer_info.append(doc.page_content)
# print(infer_info)


URI = "tcp://47.102.103.246:19530"
vector_store = Milvus(
    embeddings,
    connection_args={"uri": URI},
    collection_name="Noun_Collection_bge_v3",
)

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})


#docs = retriever.invoke("阜阳站信什么锁室内设备招标公告这个项目完整名称是什么，我不太记得")
#docs = retriever.invoke("上海城建审图咨询有限公司能否提供相关网址")
#docs = retriever.invoke("新建杭州乘务员工程施工总价承包招标公告")
docs = retriever.invoke("阜阳站信什么锁室内设备招标公告这个项目完整名称是什么，我不太记得")
# Print the retrieved documents
infer_info =[]
for doc in docs:
    #print(doc.page_content)
    #infer_info.append(doc.page_content)
    infer_info.append(doc.metadata["content"])
print(infer_info)