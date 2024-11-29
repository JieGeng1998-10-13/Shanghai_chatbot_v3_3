import re
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_milvus import Milvus
from langchain_community.embeddings import XinferenceEmbeddings
from typing import List


class XinferenceEmbeddingWrapper:
    def __init__(self, server_url: str, model_uid: str):
        # 初始化原始的 XinferenceEmbeddings
        self.embedding_model = XinferenceEmbeddings(server_url=server_url, model_uid=model_uid)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 批量嵌入文档
        return [self.embedding_model.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        # 单个查询的嵌入
        return self.embedding_model.embed_query(text)


xinference = XinferenceEmbeddingWrapper(
    server_url="http://localhost:9997", model_uid="bge-m3"
)

LAW_URI = "tcp://47.102.103.246:19530"

vector_store_clause = Milvus(
    xinference,
    connection_args={"uri": LAW_URI},
    collection_name="law_clause",
)

model = OllamaLLM(model="EntropyYue/chatglm3:6b")

template = """
根据检索到的内容，用中文回答问题：
问题：{question}
参考内容：{reference}
你的回答：
"""
prompt = PromptTemplate(
    input_variables=["question"],
    template=template,
)
chain = prompt | model


def call_model_get_law(state):
    loop_step = state.get("loop_step", 0)
    question = state["question"]
    pattern = r'《(.*?)》'
    law = re.findall(pattern, question)
    len_law = len(law)
    law_lst = []
    for i in range(len_law):
        limit_text = "law_title == \'" + law[i] + "\'"
        search_output = vector_store_clause.similarity_search(
            question,
            k=3,
            expr=limit_text,
        )
        output = [doc.page_content for doc in search_output]
        law_lst.append(search_output)

    final_answer = chain.invoke({"question": question, "reference": law_lst})
    return {"generation": final_answer, "loop_step": loop_step + 1}