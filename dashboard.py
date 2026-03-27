import gradio as gr
import pandas as pd
import os
import sqlite3
import shutil

# Settings
METADATA = "dataset/metadata.csv"
AUDIO_DIR = "dataset/audio"

def load_data(filter_status=None):
    if not os.path.exists(METADATA):
        return pd.DataFrame(columns=["path", "text", "status", "prob", "google_text", "whisper_text", "llm_text", "consensus_text"])
    try:
        df = pd.read_csv(METADATA)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame(columns=["path", "text", "status", "prob", "google_text", "whisper_text", "llm_text", "consensus_text"])
    
    if filter_status and filter_status != "All":
        df = df[df['status'] == filter_status.lower()]
    return df

def save_correction(path, new_text):
    df = pd.read_csv(METADATA)
    df.loc[df['path'] == path, 'text'] = new_text
    df.loc[df['path'] == path, 'status'] = 'corrected'
    df.to_csv(METADATA, index=False)
    return "✅ Saved!"

def delete_sample(path):
    dataset = pd.read_csv(METADATA)
    full_path = os.path.join(AUDIO_DIR, path)
    if os.path.exists(full_path):
        os.remove(full_path)
    dataset = dataset[dataset['path'] != path]
    dataset.to_csv(METADATA, index=False)
    return "🗑️ Deleted"

def get_next_sample(index, filter_status):
    dataset = load_data(filter_status)
    if index >= len(dataset):
        return None, "No more samples", "none", 0, "", "", "", "", index, ""
    
    row = dataset.iloc[index]
    audio_path = os.path.join(AUDIO_DIR, row['path'])
    return (audio_path, row['text'], row['status'], row['prob'], 
            str(row['google_text']), str(row['whisper_text']), str(row['llm_text']), str(row['consensus_text']),
            index, row['path'])

with gr.Blocks(title="Voice Dataset Reviewer (Expert Mode)") as demo:
    gr.Markdown("# 🎧 Voice Dataset Reviewer & Multi-Model Tracking")
    
    with gr.Row():
        status_filter = gr.Dropdown(["All", "Verified", "Weak", "Corrected"], label="Filter by Status", value="Weak")
        refresh_btn = gr.Button("🔄 Refresh List")

    with gr.Row():
        audio_player = gr.Audio(label="Playback")
        with gr.Column():
            text_box = gr.Textbox(label="Final Transcript (Edit to Correct)")
            status_box = gr.Label(label="Current Status")
            with gr.Row():
                google_thought = gr.Textbox(label="Google Thought", interactive=False)
                whisper_thought = gr.Textbox(label="Whisper Thought", interactive=False)
                llm_thought = gr.Textbox(label="LLM/BERT Thought", interactive=False)
            prob_box = gr.Number(label="Confidence Score")
            save_btn = gr.Button("💾 Save Correction", variant="primary")
    
    path_state = gr.State("")
    index_state = gr.State(0)
    
    with gr.Row():
        prev_btn = gr.Button("⬅️ Prev")
        del_btn = gr.Button("🗑️ Delete Sample", variant="danger")
        next_btn = gr.Button("Next Sample ➡️")

    def update_view(idx, f):
        audio, text, status, prob, g_t, w_t, l_t, c_t, current_idx, path = get_next_sample(idx, f)
        return audio, text, status, prob, g_t, w_t, l_t, current_idx, path

    save_btn.click(save_correction, inputs=[path_state, text_box], outputs=[status_box])
    
    next_btn.click(lambda i: i + 1, index_state, index_state).then(update_view, [index_state, status_filter], [audio_player, text_box, status_box, prob_box, google_thought, whisper_thought, llm_thought, index_state, path_state])
    prev_btn.click(lambda i: max(0, i - 1), index_state, index_state).then(update_view, [index_state, status_filter], [audio_player, text_box, status_box, prob_box, google_thought, whisper_thought, llm_thought, index_state, path_state])
    refresh_btn.click(lambda: 0, None, index_state).then(update_view, [index_state, status_filter], [audio_player, text_box, status_box, prob_box, google_thought, whisper_thought, llm_thought, index_state, path_state])
    del_btn.click(delete_sample, inputs=[path_state], outputs=[status_box]).then(update_view, [index_state, status_filter], [audio_player, text_box, status_box, prob_box, google_thought, whisper_thought, llm_thought, index_state, path_state])

    demo.load(update_view, [index_state, status_filter], [audio_player, text_box, status_box, prob_box, google_thought, whisper_thought, llm_thought, index_state, path_state])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
