# VaniSetu — Offline AI Voice Assistant & AAC Device

VaniSetu is an open-source, fully offline Alternative and Augmentative Communication (AAC) device. It acts as a smart, personalized voice for individuals with speech impairments. 

Built entirely to run locally on consumer hardware (as low as a 4GB VRAM GPU), it listens to ambient room context, learns the user's communication style, and generates context-aware, editable speech options in real-time.

## Key Features Built So Far
* **100% Offline & Private:** No cloud APIs. Your conversations never leave the device.
* **Always-On Listening:** Uses Silero VAD (Voice Activity Detection) to constantly monitor the room without needing to press "Record". It pauses listening automatically while speaking.
* **Temporal Ambient Memory (RAG):** Uses ChromaDB to log ambient conversations and user preferences. If Mom says "Dinner is ready," the device remembers that context for its next suggestions.
* **Lightning Fast Edge-TTS:** Uses Piper TTS running locally on the CPU to generate human-sounding voices in milliseconds.
* **Token Streaming UI:** Watch the AI generate responses word-by-word via Gradio, dramatically reducing perceived latency.
* **Hardware Optimized:** Whisper and Gemma 2 (2B SLM) are locked in VRAM for instant inference, while Chroma and Piper run on the CPU to prevent VRAM overflow.

## Design Philosophy & Architectural Trade-offs

Building an offline, real-time AAC agent requires aggressively optimizing for both low latency and strict hardware constraints. VaniSetu was intentionally designed as a **Modular Pipeline** rather than relying on a massive, end-to-end multimodal model. Here is why:

### 1. Why Not Use an End-to-End Multimodal Model (e.g., Native Audio LLMs)?
While cutting-edge models can natively process audio in and generate audio out without intermediate text, we explicitly chose to decouple the pipeline (Whisper ➔ Gemma ➔ Piper) for two critical reasons:
* **User Agency (The AAC Rule):** An end-to-end audio model bypasses the screen. For an AAC device, the user *must* have the ability to read, select, and edit the AI's intended response before it is spoken out loud. Decoupling the pipeline allows us to intercept the text generation, present 3 editable options, and keep the user in complete control.
* **Strict 4GB VRAM Budget:** Native multimodal models require massive KV caches to hold raw audio embeddings, easily overflowing older consumer GPUs. By separating the tasks, we can use an ultra-lite 2B parameter text model alongside a tiny ASR model, fitting comfortably inside 4GB of VRAM.

### 2. Silero VAD (Voice Activity Detection) vs. Continuous ASR
Running Whisper continuously on an open microphone will cause immediate thermal throttling and crash the GPU. We introduced Silero VAD on the CPU to act as a gatekeeper. It consumes less than 1MB of RAM, listens for human vocal cords, and only wakes up the GPU and Whisper when a complete sentence has been spoken, ensuring near-zero idle compute cost.

### 3. Piper TTS (CPU) vs. Cloud/GPU Audio
We replaced cloud-based engines (like Edge-TTS) with Piper TTS to guarantee 100% offline reliability. Crucially, Piper is optimized via ONNX to run entirely on the CPU. Offloading audio synthesis to the CPU reserves all available GPU VRAM strictly for Whisper (listening) and Gemma (thinking).

### 4. ChromaDB vs. FAISS for Ambient Memory
While FAISS offers microsecond advantages for pure vector math, it requires building a custom SQLite wrapper to store the actual conversational text and metadata. ChromaDB was chosen for the RAG pipeline because it acts as a complete document store, allowing us to easily tag memories with UNIX timestamps and filter out "expired" ambient context using native metadata querying.

## Roadmap & Future Vision (Contributions Welcome!)
VaniSetu is designed to be a complete "Digital Life" bridge. Here is what we are building next:

- [ ] **Onboarding Profile UI:** A dedicated tab to input Bio Data (Name, Address, Medical Info, Emergency Contacts, Food/Drink Preferences). This data will be hardcoded into the SLM's context so the device inherently knows the user without needing to ask.
- [ ] **Agentic Time-Recall:** Upgrading the RAG pipeline so the user can be asked "What did you do at 12:00 PM?" and the device can intelligently query its database based on temporal ranges.
- [ ] **MCP (Model Context Protocol) Integration:** Connecting VaniSetu to the outside world. Giving the device the ability to read emails, scroll social media, or play Spotify through secure API bridges.
- [ ] **Cloud-AI Handoff:** While the core device is offline for safety and speed, we plan to add an opt-in toggle to route complex tasks (like writing a long email or summarizing an article) to Claude or Gemini API endpoints.

## Installation & Setup
1. Clone the repo.
2. Install requirements: `pip install -r requirements.txt`
3. Download the Piper TTS voice model to the `/models` directory.
4. Run the app: `python -m src.app`