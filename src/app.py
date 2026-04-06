import gradio as gr
from src.engine import process_input_stream, log_user_choice
from src.memory import save_new_rule, distill_and_save_session
from src.utils import stream_to_speech
from src.audio_daemon import start_background_listener, audio_event_queue


# Start the always-on mic
start_background_listener()

def check_for_speech(history_state, lang_selector):
    """Checks if the VAD heard anything. If yes, it processes it."""
    if not audio_event_queue.empty():
        audio_tuple = audio_event_queue.get()
        # Create the generator for the UI
        generator = process_input_stream(
            audio_tuple=audio_tuple, 
            text_input="", 
            conversation_history=history_state, 
            modifier="None", 
            target_language=lang_selector
        )
        # Yield the results directly to the UI elements
        for result in generator:
            yield result
    else:
        # If no speech, yield a completely unchanged state
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()



def handle_audio(audio_tuple, transcript_state, history, lang):
    if audio_tuple is None:
        return transcript_state, "Waiting for audio...", "Waiting...", "Waiting...", history, "Status: 🟡 No audio received."
    
    # AUDIO LENGTH SAFEGUARD
    sample_rate, audio_array = audio_tuple
    if len(audio_array) / sample_rate > 30:
        return transcript_state, "Audio too long (max 30s).", "—", "—", history, "Status: ⚠️ Audio exceeds 30 seconds."

    transcription, o1, o2, o3, new_history = process_input_stream(
        audio_tuple, transcript_state, history, modifier=None, target_language=lang
    )
    return transcription, o1, o2, o3, new_history, "Status: 🟢 Ready"

def handle_text(text_input, history, lang):
    if not text_input or not text_input.strip():
        return text_input, "Type something above first.", "—", "—", history, "Status: ⚠️ Empty input."
    transcription, o1, o2, o3, new_history = process_input_stream(
        None, text_input, history, modifier=None, target_language=lang
    )
    return transcription, o1, o2, o3, new_history, "Status: 🟢 Ready"

def handle_modifier(modifier_label, current_transcript, history, lang):
    if not current_transcript or current_transcript == "Waiting for audio...":
        return current_transcript, "Record or type first.", "—", "—", history, "Status: ⚠️ No transcript yet."
    transcription, o1, o2, o3, new_history = process_input_stream(
        None, current_transcript, history, modifier=modifier_label, target_language=lang
    )
    return transcription, o1, o2, o3, new_history, "Status: 🟢 Ready"

def end_session(transcript_state, history):
    full_transcript = transcript_state or ""
    if history:
        lines = []
        for turn in history:
            role = turn.get("role", "")
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")
        full_transcript = "\n".join(lines)
    return distill_and_save_session(full_transcript)


def create_ui():
    import gradio as gr
    from src.engine import process_input_stream
    from src.memory import save_new_rule, distill_and_save_session
    from src.utils import stream_to_speech
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🗣️ VaniSetu — AAC Voice Assistant")
        gr.Markdown("Lightning Fast Offline Mode: Models locked in VRAM.")

        # 1. STATES
        history_state     = gr.State([])
        transcript_state  = gr.State("")
        
        # 2. UI COMPONENTS
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Input (Listen or Type context)")
                audio_in = gr.Audio(sources=["microphone"], type="numpy", streaming=False, label="🎙️ Listen to Room")
                text_in  = gr.Textbox(placeholder="Type context manually...", label="Text Context", lines=1)
                text_btn = gr.Button("▶ Generate from text context", variant="secondary")

            with gr.Column(scale=1):
                gr.Markdown("### 2. What Vani Heard")
                transcript_out = gr.Textbox(label="Transcript", interactive=False, lines=2)
                lang_selector = gr.Dropdown(choices=["English"], value="English", label="Response Language")

        gr.Markdown("### 3. Nudge AI")
        with gr.Row():
            less_btn = gr.Button("📉 Soften / Disagree")
            more_btn = gr.Button("📈 Strengthen / Expand")

        gr.Markdown("### 4. AI Suggestions (Fully Editable — Type whatever you want here)")
        with gr.Row():
            with gr.Column():
                sugg_1 = gr.Textbox(label="Option 1", lines=2)
                spk_1 = gr.Button("🔊 Speak 1", variant="primary")
            with gr.Column():
                sugg_2 = gr.Textbox(label="Option 2", lines=2)
                spk_2 = gr.Button("🔊 Speak 2", variant="primary")
            with gr.Column():
                sugg_3 = gr.Textbox(label="Option 3", lines=2)
                spk_3 = gr.Button("🔊 Speak 3", variant="primary")

        audio_out   = gr.Audio(label="Voice Output", autoplay=True, visible=False)
        status_bar  = gr.Markdown("Status: 🟢 System Ready")

        with gr.Row():
            save_btn       = gr.Button("💾 Remember this choice", variant="secondary")
            end_session_btn= gr.Button("⏹ End Session + Save Memory", variant="secondary")

        # Timer Component
        timer = gr.Timer(1)

        # 3. WIRING (Everything goes down here!)
        
        # Timer Wiring
        timer.tick(
            check_for_speech,
            inputs=[history_state, lang_selector],
            outputs=[transcript_out, sugg_1, sugg_2, sugg_3, history_state]
        )

        # Input Processing Wiring
        audio_in.stop_recording(
            process_input_stream,
            inputs=[audio_in, text_in, history_state, gr.State("None"), lang_selector],
            outputs=[transcript_out, sugg_1, sugg_2, sugg_3, history_state]
        )
        text_btn.click(
            process_input_stream,
            inputs=[gr.State(None), text_in, history_state, gr.State("None"), lang_selector],
            outputs=[transcript_out, sugg_1, sugg_2, sugg_3, history_state]
        )
        
        less_btn.click(
            process_input_stream,
            inputs=[gr.State(None), transcript_state, history_state, gr.State("Soften / express less forcefully"), lang_selector],
            outputs=[transcript_out, sugg_1, sugg_2, sugg_3, history_state]
        )
        more_btn.click(
            process_input_stream,
            inputs=[gr.State(None), transcript_state, history_state, gr.State("Strengthen / expand on the point"), lang_selector],
            outputs=[transcript_out, sugg_1, sugg_2, sugg_3, history_state]
        )

        # The Speak buttons remain tied exclusively to user clicks!
        from src.engine import log_user_choice # Ensure this is imported to log the preference!
        
        spk_1.click(stream_to_speech, inputs=[sugg_1, lang_selector]).then(log_user_choice, inputs=[transcript_out, sugg_1])
        spk_2.click(stream_to_speech, inputs=[sugg_2, lang_selector]).then(log_user_choice, inputs=[transcript_out, sugg_2])
        spk_3.click(stream_to_speech, inputs=[sugg_3, lang_selector]).then(log_user_choice, inputs=[transcript_out, sugg_3])
        
        save_btn.click(save_new_rule, inputs=[transcript_out, sugg_1], outputs=status_bar)
        
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(debug=True, share=True)