import gradio as gr
import pandas as pd
import os
import sqlite3
import shutil

# Settings
METADATA = "dataset/metadata.csv"
AUDIO_DIR = "dataset/audio"

def load_data():
    if not os.path.exists(METADATA):
        return pd.DataFrame(columns=["path", "text", "prob", "consensus_verified", "llm_verified"])
    return pd.read_csv(METADATA)

def delete_sample(path):
    dataset = load_data()
    # Delete file
    full_path = os.path.join(AUDIO_DIR, path)
    if os.path.exists(full_path):
        os.remove(full_path)
    # Remove from CSV
    dataset = dataset[dataset['path'] != path]
    dataset.to_csv(METADATA, index=False)
    return dataset

def get_next_sample(index):
    dataset = load_data()
    if index >= len(dataset):
        return None, "No more samples", 0, False, False, index
    
    row = dataset.iloc[index]
    audio_path = os.path.join(AUDIO_DIR, row['path'])
    return audio_path, row['text'], row['prob'], row['consensus_verified'], row['llm_verified'], index

with gr.Blocks(title="Voice Dataset Reviewer") as demo:
    gr.Markdown("# 🎧 Voice Dataset Reviewer")
    gr.Markdown("Review, listen, and clean your 24/7 background collected dataset.")
    
    with gr.Row():
        audio_player = gr.Audio(label="Word Segment")
        with gr.Column():
            text_box = gr.Textbox(label="Transcript")
            prob_box = gr.Number(label="AI Confidence")
            consensus_box = gr.Checkbox(label="Engine Consensus", interactive=False)
            llm_box = gr.Checkbox(label="LLM Verified", interactive=False)
    
    index_state = gr.State(0)
    
    with gr.Row():
        prev_btn = gr.Button("⬅️ Prev")
        del_btn = gr.Button("🗑️ Delete", variant="danger")
        next_btn = gr.Button("Next ➡️", variant="primary")

    def update_view(idx):
        audio, text, prob, cons, llm, current_idx = get_next_sample(idx)
        return audio, text, prob, cons, llm, current_idx

    next_btn.click(lambda i: i + 1, index_state, index_state).then(update_view, index_state, [audio_player, text_box, prob_box, consensus_box, llm_box, index_state])
    prev_btn.click(lambda i: max(0, i - 1), index_state, index_state).then(update_view, index_state, [audio_player, text_box, prob_box, consensus_box, llm_box, index_state])
    del_btn.click(delete_sample, inputs=[gr.State(value="")], outputs=[]).then(lambda i: i, index_state, index_state).then(update_view, index_state, [audio_player, text_box, prob_box, consensus_box, llm_box, index_state])

    # Initial load
    demo.load(update_view, index_state, [audio_player, text_box, prob_box, consensus_box, llm_box, index_state])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
