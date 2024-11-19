from Figure_search.app_figure_search import search_image
from chatbot.app_gradio import answer_question
from chatbot.example_list import examples
import gradio as gr

# Combined Gradio Interface
with gr.Blocks() as iface:
    gr.Markdown("## 上海市交易系统问答平台与商品搜索（Chatglm3）")
    # Question-Answering Section
    gr.Markdown("### 问答功能")
    question_input = gr.Textbox(label="请输入问题", lines=2, interactive=True)
    question_button = gr.Button("提交问题")

    gr.Markdown("#### 示例问题")
    gr.Examples(examples=examples, inputs=question_input, label=None)
    with gr.Row():
        answer_output = gr.Textbox(label="回答", lines=5)
    question_button.click(answer_question, inputs=question_input, outputs=answer_output)

    # Image Retrieval Section
    gr.Markdown("### 商品图片检索功能")
    with gr.Group():
        with gr.Row():
            image_input = gr.Image(type="numpy", label="上传商品图片")
        with gr.Row():
            image_output = gr.Gallery(label="相似图片")
        with gr.Row():
            paths_output = gr.Textbox(label="商品型号", lines=10)
        image_input.change(search_image, inputs=image_input, outputs=[image_output, paths_output])
# 启动应用程序
if __name__ == "__main__":
    iface.launch(share=True)