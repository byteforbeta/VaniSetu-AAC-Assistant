# VaniSetu — Offline AI Voice Assistant & AAC Software for Edge Devices

VaniSetu is an open-source, fully offline Alternative and Augmentative Communication (AAC) device. It acts as a smart, personalized voice for individuals with speech impairments. 

Built entirely to run locally on consumer hardware (as low as a 4GB VRAM GPU), it listens to ambient room context, learns the user's communication style, and generates context-aware, editable speech options in real-time.

## Why VaniSetu? (The AAC Paradigm Shift)

Traditional AAC devices and apps suffer from major limitations that VaniSetu actively solves:

1. **The "Grid" Bottleneck:** Current market leaders rely on nested folders of icons. If someone asks, "Do you want to go to the park?", the user has to manually navigate through `Places -> Outdoors -> Park -> Yes` to reply. **VaniSetu is Context-Aware.** Because it listens to the ambient room, it pre-generates contextually relevant answers *before* the user even touches the screen.
2. **Cloud Dependency & Privacy:** Modern AI voice tools are incredible, but they require active internet connections and expensive API subscriptions. If a user loses cell service, they literally lose their voice. Furthermore, beaming intimate, daily conversations to corporate cloud servers is a massive privacy risk. **VaniSetu is 100% Offline.**
3. **Prohibitive Hardware Costs:** Dedicated AAC hardware can cost thousands of dollars. VaniSetu is engineered to run flawlessly on standard, accessible consumer hardware—requiring as little as a 4GB VRAM GPU to run the entire pipeline in real-time.
4. **Lack of Personalization:** Most devices sound robotic and use static phrases. VaniSetu learns the user's specific communication style via a dual-track Temporal RAG system, adapting its suggestions to match how the user *actually* prefers to speak.

## ❤️ The Mission: Defeating the "2-Minute Delay"

In the world of AAC, latency is not just a technical metric; it is an accessibility barrier. Traditional grid-based AAC devices often impose a severe time penalty on the user—frequently creating a 1 to 2-minute delay between a question being asked and the user finishing their typed response. 

This delay has devastating real-world consequences:

* **The "Caregiver Bypass":** Because of the uncomfortable silence and delay, strangers, doctors, and even friends often give up waiting and instinctively start speaking *about* the user to their caretaker, rather than speaking *to* the user. This strips the user of their autonomy and humanity.
* **The Psychological Toll:** Imagine knowing exactly what you want to say, but being trapped behind a 2-minute typing wall while the conversation completely moves on without you. This extreme frustration frequently leads to panic, shouting, crying, or emotional outbursts. These are not behavioral issues; they are the desperate symptoms of being trapped by latency.
* **The Loss of Confidence:** When every interaction feels like a terrifying race against someone else's patience, users lose the confidence to speak up at all, eventually withdrawing from social interaction.

**Speed is Inclusion.** We built VaniSetu's unique architecture—Ambient RAG pre-generation, token streaming, and local TTS—specifically to eradicate this delay. By listening to the room and generating context-aware options *while* the other person is still talking, VaniSetu reduces a 2-minute struggle to a 2-second click. 

This project isn't just about playing with LLMs; it is about keeping people in the conversation, restoring their confidence, and ensuring they are never spoken over again.

## Key Features Built So Far

* **100% Offline & Private:** No cloud APIs. Your conversations never leave the device.
* **Always-On Listening:** Uses Silero VAD (Voice Activity Detection) to constantly monitor the room without needing to press "Record". It pauses listening automatically while the device is speaking.
* **Temporal Ambient Memory (RAG):** Uses ChromaDB to log ambient conversations and user preferences. If someone says "Dinner is ready," the device remembers that context for its next suggestions.
* **Lightning Fast Edge-TTS:** Uses Piper TTS running locally on the CPU to generate human-sounding voices in milliseconds.
* **Token Streaming UI:** Watch the AI generate responses word-by-word via Gradio, dramatically reducing perceived latency.

## Design Philosophy & Architectural Trade-offs

Building an offline, real-time AAC agent requires aggressively optimizing for both low latency and strict hardware constraints. VaniSetu was intentionally designed as a **Modular Pipeline** rather than relying on a monolithic multimodal model. Here is why:

### 1. Why Not Use an End-to-End Multimodal Model?
While cutting-edge models can natively process audio in and generate audio out without intermediate text, we explicitly chose to decouple the pipeline (Whisper -> Gemma -> Piper) for two critical reasons:
* **User Agency (The AAC Rule):** An end-to-end audio model bypasses the screen. For an AAC device, the user *must* have the ability to read, select, and edit the AI's intended response before it is spoken out loud. Decoupling the pipeline allows us to intercept the text generation, present 3 editable options, and keep the user in complete control.
* **Strict 4GB VRAM Budget:** Native multimodal models require massive KV caches to hold raw audio embeddings, easily overflowing older consumer GPUs. By separating the tasks, we can use an ultra-lite 2B parameter text model alongside a tiny ASR model, fitting comfortably inside 4GB of VRAM.

### 2. Silero VAD vs. Continuous ASR
Running Whisper continuously on an open microphone will cause immediate thermal throttling and crash the GPU. We introduced Silero VAD on the CPU to act as a gatekeeper. It consumes less than 1MB of RAM, listens for human vocal cords, and only wakes up the GPU and Whisper when a complete sentence has been spoken, ensuring near-zero idle compute cost.

### 3. Piper TTS (CPU) vs. Cloud/GPU Audio
We replaced cloud-based engines with Piper TTS to guarantee 100% offline reliability. Crucially, Piper is optimized via ONNX to run entirely on the CPU. Offloading audio synthesis to the CPU reserves all available GPU VRAM strictly for Whisper (listening) and Gemma (thinking).

### 4. ChromaDB vs. FAISS for Ambient Memory
While FAISS offers microsecond advantages for pure vector math, it requires building a custom SQLite wrapper to store the actual conversational text and metadata. ChromaDB was chosen for the RAG pipeline because it acts as a complete document store, allowing us to easily tag memories with UNIX timestamps and filter out "expired" ambient context using native metadata querying.

## Roadmap & Future Vision (Contributions Welcome!)

VaniSetu is designed to be a complete "Digital Life" bridge. Here is what we are building next:

- [ ] **Onboarding Profile UI:** A dedicated tab to input Bio Data (Name, Address, Medical Info, Emergency Contacts, Food/Drink Preferences). This data will be hardcoded into the SLM's context so the device inherently knows the user without needing to ask.
- [ ] **Agentic Time-Recall:** Upgrading the RAG pipeline so the user can be asked "What did you do at 12:00 PM?" and the device can intelligently query its database based on temporal ranges.
- [ ] **MCP (Model Context Protocol) Integration:** Connecting VaniSetu to the outside world. Giving the device the ability to read emails, scroll social media, or play music through secure API bridges.
- [ ] **Cloud-AI Handoff:** While the core device is offline for safety and speed, we plan to add an opt-in toggle to route complex tasks (like writing a long email or create music etc) to more powerful API endpoints.

## Installation & Local Setup

Because VaniSetu runs entirely offline, you need to download the open-source AI models to your local machine before running the app.

### Step 1: Clone and Install
```bash
git clone [https://github.com/byteforbeta/VaniSetu-AAC-Assistant.git](https://github.com/byteforbeta/VaniSetu-AAC-Assistant.git)
cd VaniSetu-AAC-Assistant
pip install -r requirements.txt
```
 
### Step 2: Set up the models Directory
Create a folder named `models` inside the main project directory.
```bash
mkdir models
```

### Step 3: Download the Language Model (Gemma)
We use a quantized version of Gemma 2 (2B parameters) to ensure it fits perfectly into 4GB of VRAM while maintaining high conversational intelligence.
1. Go to HuggingFace (e.g., search for `bartowski/gemma-2-2b-it-GGUF`).
2. Download a `Q4_K_M` or `Q5_K_M` `.gguf` file (usually around 1.5GB to 2GB).
3. Place the `.gguf` file inside your new `models/` folder.
4. *Note: Update `src/config.py` to point to the exact filename you downloaded.*

### Step 4: Download the Voice Model (Piper TTS)
We use Piper for instant, offline voice synthesis.
1. Go to the Piper Voices Repository on HuggingFace.
2. Choose a voice (we recommend `en_US-lessac-medium`).
3. Download exactly two files: the `.onnx` file and the `.onnx.json` file.
4. Place both files inside your `models/` folder.

### Step 5: Run the Device
Once your models are in place, start the VaniSetu interface:
```bash
python -m src.app
```