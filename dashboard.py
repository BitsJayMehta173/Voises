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
        return pd.DataFrame(columns=["path", "text_roman", "text_native", "status", "prob", "google_text", "whisper_roman", "whisper_native", "llm_text"])
    try:
        df = pd.read_csv(METADATA)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame(columns=["path", "text_roman", "text_native", "status", "prob", "google_text", "whisper_roman", "whisper_native", "llm_text"])
    
    if filter_status and filter_status != "All":
        df = df[df['status'] == filter_status.lower()]
    return df

def save_correction(path, new_text):
    df = pd.read_csv(METADATA)
    df.loc[df['path'] == path, 'text_roman'] = new_text
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
        return None, "No more samples", "", "none", 0, "", "", "", index, ""
    
    row = dataset.iloc[index]
    audio_path = os.path.join(AUDIO_DIR, row['path'])
    return (audio_path, row['text_roman'], str(row.get('text_native', '')), row['status'], row['prob'], 
            str(row.get('google_text', '')), str(row.get('whisper_roman', '')), str(row.get('whisper_native', '')),
            index, row['path'])

with gr.Blocks(title="Voice Dataset Reviewer (Expert Mono-Script Mode)") as demo:
    gr.Markdown("# 🎧 Voice Dataset Reviewer (Devnagri Support)")
    
    with gr.Row():
        status_filter = gr.Dropdown(["All", "Verified", "Weak", "Corrected"], label="Filter by Status", value="Weak")
        refresh_btn = gr.Button("🔄 Refresh List")

    with gr.Row():
        audio_player = gr.Audio(label="Playback")
        with gr.Column():
            with gr.Row():
                text_box_roman = gr.Textbox(label="Romanized Text (mera, naam)")
                text_box_native = gr.Textbox(label="Native Script (नाम, मेरा)")
            status_box = gr.Label(label="Current Status")
            with gr.Row():
                google_thought = gr.Textbox(label="Google Thought", interactive=False)
                whisper_thought = gr.Textbox(label="Whisper Roman", interactive=False)
                whisper_native_thought = gr.Textbox(label="Whisper Native", interactive=False)
            prob_box = gr.Number(label="Confidence Score")
            save_btn = gr.Button("💾 Save Correction (Romanized)", variant="primary")
    
    path_state = gr.State("")
    index_state = gr.State(0)
    
    with gr.Row():
        prev_btn = gr.Button("⬅️ Prev")
        del_btn = gr.Button("🗑️ Delete Sample", variant="danger")
        next_btn = gr.Button("Next Sample ➡️")

    def update_view(idx, f):
        audio, text_r, text_n, status, prob, g_t, w_r, w_n, current_idx, path = get_next_sample(idx, f)
        return audio, text_r, text_n, status, prob, g_t, w_r, w_n, current_idx, path

    save_btn.click(save_correction, inputs=[path_state, text_box_roman], outputs=[status_box])
    
    components = [audio_player, text_box_roman, text_box_native, status_box, prob_box, google_thought, whisper_thought, whisper_native_thought, index_state, path_state]
    next_btn.click(lambda i: i + 1, index_state, index_state).then(update_view, [index_state, status_filter], components)
    prev_btn.click(lambda i: max(0, i - 1), index_state, index_state).then(update_view, [index_state, status_filter], components)
    refresh_btn.click(lambda: 0, None, index_state).then(update_view, [index_state, status_filter], components)
    del_btn.click(delete_sample, inputs=[path_state], outputs=[status_box]).then(update_view, [index_state, status_filter], components)

    demo.load(update_view, [index_state, status_filter], components)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
