import json
import os
from langgraph.graph import StateGraph
import gradio as gr
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tools import GraphState  # 图结构数据类型声明
from langgraph.graph import END
from utils.function_tools import route_question, call_model, call_model_SQL_prompt
from utils.function_tools import call_model_SQL, law_question_prompt, call_model_SQL_enhence
from utils.function_tools import call_model_filter
from utils.rag_text import call_model_raglaw
from langgraph.checkpoint.memory import MemorySaver
from utils.enhencement_functions import route_question_enhencement
#from utils.noun_retriever_Mil import noun_retriever_prompt
from utils.noun_retriever import noun_retriever_prompt
from utils.law_selection import route_question_law


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("answer_directly", call_model)  # web search
workflow.add_node("retrieve_SQL_prompt", call_model_SQL_prompt)
# workflow.add_node("retrieve_SQL", call_model_SQL_vanna)  # retrieve， 第一种vanna方案
workflow.add_node("retrieve_SQL", call_model_SQL)  # 第二种SQL查询方案，用langchain自带方法,方案切换只需要注释掉其中一个节点即可

# workflow.add_node("SQL_enhencement", label_predict_SQL)
workflow.add_node("SQL_enhencement", noun_retriever_prompt)
#workflow.add_node("enhencement_processing", call_model_SQL)
workflow.add_node("enhencement_processing", call_model_SQL_enhence)
workflow.add_node("information_filter", call_model_filter)


#Define four law nodes

# Define three four nodes
workflow.add_node("law_question", law_question_prompt) # 替换为正确的函数

workflow.add_node("retrieve_law", call_model_raglaw)
workflow.add_node("meaning_large_retrieve", call_model_raglaw) #替换为正确的函数
workflow.add_node("similarity_retrieve", call_model_raglaw)

# Define the edges
workflow.add_edge("answer_directly", END)
# workflow.add_edge("retrieve_SQL_prompt", "retrieve_SQL")
workflow.add_edge("retrieve_SQL", END)
workflow.add_edge("retrieve_law", END)
workflow.add_edge("SQL_enhencement", "enhencement_processing")
workflow.add_edge("enhencement_processing", "information_filter")
workflow.add_edge("information_filter", END)


# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "answer_directly": "answer_directly",
        "vectorstore": "retrieve_SQL_prompt",
        "law_query": "law_question",
    },
)


workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "retrieve_SQL_prompt",
    # Next, we pass in the function that will determine which node is called next.
    route_question_enhencement,
    {
        "no_enhencement": "retrieve_SQL",
        "enhencement": "SQL_enhencement"
    },
)


workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "law_question",
    # Next, we pass in the function that will determine which node is called next.
    route_question_law,
    {
        "clear_question": "retrieve_law",
        "law_details": "meaning_large_retrieve",
        "law_examples": "similarity_retrieve"

    },
)


memory = MemorySaver()
graph = workflow.compile()

# 读取 JSON 文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 写入 JSON 文件
def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 修改后的 answer_question 函数
def answer_question(question):
    inputs = {"question": question}
    outputs = None
    # 运行图并获取最终输出
    for event in graph.stream(inputs, stream_mode="values"):
        outputs = event  # 假设最后一个事件是最终输出
    # 提取 'generation' 字段
    generation = outputs.get("generation", "未生成回答")
    return generation

# 处理 JSON 文件并更新答案
def process_json(input_file, output_file):
    # 加载 JSON 数据
    data = load_json(input_file)
    
    # 遍历每个问题并调用 answer_question 获取答案
    for entry in data:
        question = entry.get("Question")
        if question:
            app_answer = answer_question(question)
            # 将生成的答案添加到字典中
            entry["app_Answer"] = app_answer
    
    # 保存更新后的 JSON 数据
    save_json(output_file, data)

# 示例：调用处理函数
input_json_file = "/mnt/workspace/上海市交通系统交易问答框架/评测sql脚本数据集/questions.json"  # 输入的 JSON 文件路径
output_json_file = "/mnt/workspace/上海市交通系统交易问答框架/评测sql脚本数据集/questions_with_answers.json"  # 输出的 JSON 文件路径

if __name__ == "__main__":
    process_json(input_json_file, output_json_file)
