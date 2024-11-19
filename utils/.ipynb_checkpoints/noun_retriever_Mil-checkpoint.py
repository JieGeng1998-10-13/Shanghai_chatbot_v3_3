from langchain_ollama import  OllamaEmbeddings
from langchain_milvus import Milvus


def noun_retriever_prompt(state):
    question = state["question"]
    # Initialize embeddings and vector store
    embeddings = OllamaEmbeddings(model="bge-m3:latest")
    URI = "tcp://47.102.103.246:19530"
    vector_store = Milvus(
        embeddings,
        connection_args={"uri": URI},
        #collection_name="NounCollection2",
        collection_name="Noun_Collection_bge_v3",
    )

    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})
    docs = retriever.invoke(question)
    # Print the retrieved documents
    infer_info =[]
    prompt = """
             结合原本问题和检索结果回答问题，
             你要根据问题选出合适的结果回答
             """
    prompt_enhence = """
             结合原本问题和检索结果回答问题，
             每次查询都有四个结果放在列表中,
             按照和问题的相似度从高到低排序，
             每个检索结果被逗号隔开，
             你要根据问题选出合适的结果回答，
             如果要用名称进行检索，请选择合适的正确的名称
             """
    for doc in docs:
        #print(doc.page_content)
        infer_info.append(doc.metadata["content"])
    new_question = "原本问题是：" + question + "现在经过检索器后得到下列结果" + str(infer_info) + prompt
    return {"question": new_question}


