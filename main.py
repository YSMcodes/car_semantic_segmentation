import gradio as gr
from src.app.interface_main import create_interface

demo = create_interface()

if __name__ == "__main__":
    demo.launch()