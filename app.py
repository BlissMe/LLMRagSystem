import gradio as gr
from main import app  # import your FastAPI app

# This function is just to keep HF Space alive and give a public URL
def check_status():
    return "FastAPI app is running!"

# Minimal Gradio interface
iface = gr.Interface(fn=check_status, inputs=[], outputs="text")

# Launch Gradio on default HF Space server port
iface.launch(server_name="0.0.0.0", server_port=7860)
