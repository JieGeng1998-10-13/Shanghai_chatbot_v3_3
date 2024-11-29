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

vector_store_title = Milvus(
    xinference,
    connection_args={"uri": LAW_URI},
    collection_name="law_title",
)

retriever_title = vector_store_title.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 3})
retriever_clause = vector_store_clause.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 3})

model_json = OllamaLLM(model="EntropyYue/chatglm3:6b", format="json")
model_str = OllamaLLM(model="EntropyYue/chatglm3:6b")

import re

def cut_question(text):
    pattern = r"\d\.\s(.*?)(?=\n\d\.|\Z)"
    results = re.findall(pattern, text, re.DOTALL)
    output_list = [result.strip() for result in results]
    return output_list


template1 = """
你是一个问题分解助理，你需要做的是把问题分解成各个任务发给其他人，请按照格式返回。
请根据以下规则进行问题分解，返回你对下面部门的人的命令。
注意：你只需要考虑怎么分配任务，不用想怎么解决问题。
问题：{question}
规则： 1. 按维度拆分问题
            目标：将问题从多个维度进行分解，确保覆盖全面。

            常见维度：
                法律维度：请写出应该去查哪方面的法律
                责任维度：请写出谁应为这些行为负责
                解决维度：请写出要解决什么问题
返回形式：
    {{"法律维度": *****, "责任维度": *****, "解决维度": *****}}
    在 ***** 填入你的回答
接下来，请回答问题：
你的回答：
"""

template2 = """
请根据以下问题和分析维度，将问题分解为若干个具体的子问题，用中文回答：

问题：{question}

分析维度：{dimension_content}

请根据以下问题和分析维度，将问题分解为若干具体的子问题：


要求：
1. 每个维度列出 2-3 个具体子问题，避免层次嵌套。
2. 子问题应简明扼要。
3. 输出格式：
      法律维度：
      1. 子问题
      2. 子问题

   责任维度：
      1. 子问题
      2. 子问题

   解决维度：
      1. 子问题
      2. 子问题
最终输出为以下格式：
     [子问题, 子问题, 子问题, 子问题, 子问题, 子问题]
接下来，请回答问题：
你的回答：
"""
template3 = """
请根据参考和问题，通过参考提供的不同角度，用中文回答问题
问题：{question}
参考：{reference}
接下来，请回答问题：
你的回答：
"""

prompt1 = PromptTemplate(
    input_variables=["question"],
    template=template1,
)
chain1 = prompt1 | model_json

prompt2 = PromptTemplate(
    input_variables=["question", "dimension_content"],
    template=template2,
)
chain2 = prompt2 | model_str

prompt3 = PromptTemplate(
    input_variables=["question", "reference"],
    template=template3,
)
chain3 = prompt3 | model_str


def choose_title(lst):
    important_law = ["中华人民共和国招标投标法实施条例",
                     "中华人民共和国招标投标法",
                     "中华人民共和国政府采购法"]
    mid_important_law = ["中华人民共和国政府采购法实施条例",
                         "政府采购货物和服务招标投标管理办法",
                         "机电产品国际招标投标实施办法（试行）",
                         "必须招标的工程项目规定",
                         "工程建设项目施工招标投标办法",
                         "政府采购非招标采购方式管理办法",
                         "工程建设项目货物招标投标办法"]
    for title in lst:
        if title in important_law:
            return (0, important_law.index(title))  # 第一批，保持 list1 中的顺序
        elif title in mid_important_law:
            return (1, mid_important_law.index(title))  # 第二批，保持 list2 中的顺序
        else:
            return (2, lst.index(title))  # 其他的，保持原始顺序
        return sorted(titles, key=sort_key)


def call_problem_detail(state):
    print("函数正常调用了")
    loop_step = state.get("loop_step", 0)
    question = state["question"]
    # 分子问题
    dimension_content = chain1.invoke({"question": question})
    print("第一次invoke成功")
    question_str = chain2.invoke({"question": question, "dimension_content": dimension_content})
    print("第二次invoke成功")
    question_lst = cut_question(question_str)
    # 检索子问题
    len_lst = len(question_lst)
    reference_lst = []
    for i in range(len_lst):
        q = question_lst[i]
        title_retrieve = retriever_title.invoke(q)
        title_lst = [doc.page_content for doc in title_retrieve]
        title_choose = choose_title(title_lst)
        law_retrieve = retriever_clause.invoke(title_choose)
        law_lst = [doc.page_content for doc in law_retrieve]
        print(q, law_lst)
        reference_lst.append({"sub_q": q, "reference": law_lst})
    # 回答问题
    final_answer = chain3.invoke({"question": question, "reference": reference_lst})
    print("第三次invoke成功")
    return {"generation": final_answer, "loop_step": loop_step + 1}

# def call_problem_detail(state):
#     print("函数正常调用了")
#     loop_step = state.get("loop_step", 0)
#     final_answer = "什么情况"
#     return {"final_answer": final_answer, "loop_step": loop_step + 1}

if __name__ == "__main__":
    print(call_problem_detail({"question": "加分项是否会存在排斥潜在投标人的嫌疑?", "loop_step": 0}))