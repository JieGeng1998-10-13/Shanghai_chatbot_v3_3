import re
def route_question_law(state):
    question = state["question"]
    law_title = r'《.*?》'
    length = 50
    if len(question) > length:
        if re.search(law_title, question):
            print("---正在把问题引导至某条法律查询---")
            return "get_law"
        else:
            print("---正在把问题引导至案例查询---")
            return "law_examples"
    else:
        print("---正在把问题引导至法律细节查询---")
        return "law_details"

