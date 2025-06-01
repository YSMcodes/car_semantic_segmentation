import gradio as gr
from ..process.process import pipeline

def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("Segment Car Street")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Upload Image", type="numpy")
            with gr.Column():
                image_output = gr.Image(label="Processed Image", type="numpy")

        image_input.change(fn=pipeline, inputs=image_input, outputs=image_output)

    return demo


