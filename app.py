import os
import uuid
import json
import tempfile
import base64

import dotenv

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# --- Audio libs ---
import assemblyai as aai
from elevenlabs.client import ElevenLabs


# ==========================
#   ENV + BASE SETUP
# ==========================

dotenv.load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not ASSEMBLYAI_API_KEY:
    raise RuntimeError("Missing ASSEMBLYAI_API_KEY in .env")
if not ELEVENLABS_API_KEY:
    raise RuntimeError("Missing ELEVENLABS_API_KEY in .env")

# Configure SDKs
aai.settings.api_key = ASSEMBLYAI_API_KEY
eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# LangChain LLM
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# SQL DB
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

print("Loaded SQL tools:")
for tool in tools:
    print(f"- {tool.name}: {tool.description}")

system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.
Then you should query the schema of the most relevant tables.

do not answer any mathematical calculations or any question related to famous
celebrities or historical events.
""".format(
    dialect=db.dialect,
    top_k=5,
)

agent = create_agent(
    model,
    tools,
    system_prompt=system_prompt,
    checkpointer=InMemorySaver(),
)


# ==========================
#   HELPER: EXTRACT TEXT
# ==========================

def extract_text_from_message(message):
    """Return a plain text string from a LangChain message.content."""
    content = message.content

    # Case 1: model returned a simple string
    if isinstance(content, str):
        return content

    # Case 2: model returned a list of content blocks (Gemini style)
    if isinstance(content, list):
        parts = []
        for block in content:
            # dict style: {'type': 'text', 'text': '...'}
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(block["text"])
            # object style: e.g. has .text attribute
            elif hasattr(block, "text"):
                parts.append(block.text)

        if parts:
            return "".join(parts)

    # Fallback – at least don't crash
    return str(content)


# ==========================
#   FASTAPI APP
# ==========================

app = FastAPI(title="SQL Chat + Voice Backend")

# CORS so you can call from a frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in prod, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "ok", "message": "SQL chat + voice backend running"}


# ==========================
#   TEXT CHAT VIA WEBSOCKET
# ==========================

@app.websocket("/ws/sql-chat")
async def sql_chat(ws: WebSocket):
    """
    WebSocket endpoint for text-only chat with the SQL agent.

    Protocol:
    - Client sends plain text messages (user questions).
    - Server responds with JSON:
        { "type": "answer", "content": "<assistant response>" }
      or
        { "type": "error", "content": "<error message>" }
    """
    await ws.accept()

    # a unique conversation id for this websocket connection
    thread_id = str(uuid.uuid4())
    print(f"[WS] New connection with thread_id={thread_id}")

    try:
        while True:
            # receive user's text message
            user_query = await ws.receive_text()
            user_query = user_query.strip()

            if not user_query:
                # ignore empty messages
                continue

            print(f"[WS][{thread_id}] User: {user_query}")

            # call your SQL agent
            try:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": user_query}]},
                    {"configurable": {"thread_id": thread_id}},
                )
                ai_message = result["messages"][-1]
                answer_text = extract_text_from_message(ai_message)

                print(f"[WS][{thread_id}] Assistant: {answer_text}")

                await ws.send_json(
                    {
                        "type": "answer",
                        "content": answer_text,
                    }
                )

            except Exception as e:
                err_msg = f"Agent error: {e}"
                print(f"[WS][{thread_id}] {err_msg}")
                await ws.send_json({"type": "error", "content": err_msg})

    except WebSocketDisconnect:
        print(f"[WS] Client disconnected (thread_id={thread_id})")
    except Exception as e:
        print(f"[WS] Unexpected error: {e}")


# ==========================
#   VOICE VIA WEBSOCKET
# ==========================

@app.websocket("/ws/sql-voice")
async def sql_voice(ws: WebSocket):
    """
    WebSocket endpoint for voice queries.

    Protocol:
    - Client sends JSON text frames:
        {"type": "start_voice"}
      then binary frames with audio chunks (webm/opus),
      then JSON text frame:
        {"type": "stop_voice"}

    - Server buffers audio between start/stop,
      runs STT (AssemblyAI) -> SQL agent -> TTS (ElevenLabs),
      sends back one JSON message:
        {
          "type": "voice_result",
          "transcript": "...",
          "answer": "...",
          "audio_base64": "<mp3-base64>",
          "audio_mime": "audio/mpeg"
        }
    """
    await ws.accept()
    session_id = str(uuid.uuid4())
    print(f"[VOICE][{session_id}] WebSocket connected")

    recording = False
    audio_buffer = bytearray()

    try:
        while True:
            msg = await ws.receive()

            # Handle disconnect
            if msg["type"] == "websocket.disconnect":
                print(f"[VOICE][{session_id}] Disconnected")
                break

            # Binary audio data from client
            if msg.get("bytes") is not None:
                if recording:
                    audio_buffer.extend(msg["bytes"])
                continue

            # Text frame (JSON control)
            if msg.get("text") is not None:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "Invalid JSON"})
                    continue

                mtype = data.get("type")

                if mtype == "start_voice":
                    print(f"[VOICE][{session_id}] start_voice")
                    recording = True
                    audio_buffer = bytearray()

                elif mtype == "stop_voice":
                    print(f"[VOICE][{session_id}] stop_voice (size={len(audio_buffer)} bytes)")
                    recording = False

                    if not audio_buffer:
                        await ws.send_json({
                            "type": "error",
                            "message": "No audio received.",
                        })
                        continue

                    # ---- 1) Save audio to a temp .webm file ----
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                        tmp.write(audio_buffer)
                        tmp_path = tmp.name

                    # ---- 2) Transcribe with AssemblyAI ----
                    try:
                        transcriber = aai.Transcriber()
                        tr = transcriber.transcribe(tmp_path)

                        if tr.error:
                            raise RuntimeError(tr.error)

                        transcript_text = (tr.text or "").strip()
                    except Exception as e:
                        err = f"AssemblyAI STT error: {e}"
                        print(f"[VOICE][{session_id}] {err}")
                        await ws.send_json({"type": "error", "message": err})
                        continue

                    if not transcript_text:
                        await ws.send_json({
                            "type": "error",
                            "message": "Could not detect speech in the audio.",
                        })
                        continue

                    print(f"[VOICE][{session_id}] Transcript: {transcript_text}")

                    # ---- 3) Call SQL agent with transcript ----
                    try:
                        result = agent.invoke(
                            {"messages": [{"role": "user", "content": transcript_text}]},
                            {"configurable": {"thread_id": session_id}},
                        )
                        ai_message = result["messages"][-1]
                        answer_text = extract_text_from_message(ai_message).strip()
                    except Exception as e:
                        err = f"SQL agent error: {e}"
                        print(f"[VOICE][{session_id}] {err}")
                        await ws.send_json({"type": "error", "message": err})
                        continue

                    print(f"[VOICE][{session_id}] Answer: {answer_text}")

                    # ---- 4) TTS with ElevenLabs ----
                    try:
                        tts_stream = eleven_client.text_to_speech.convert(
                            text=answer_text or "Sorry, I could not generate an answer.",
                            voice_id="JBFqnCBsd6RMkjVDRZzb",  # change to your preferred voice
                            model_id="eleven_multilingual_v2",
                            output_format="mp3_44100_128",
                        )
                        audio_bytes = b"".join(chunk for chunk in tts_stream)
                        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                    except Exception as e:
                        err = f"ElevenLabs TTS error: {e}"
                        print(f"[VOICE][{session_id}] {err}")
                        # still send text result even if TTS fails
                        await ws.send_json({
                            "type": "voice_result",
                            "transcript": transcript_text,
                            "answer": answer_text,
                            "audio_base64": None,
                            "audio_mime": None,
                            "warning": err,
                        })
                        continue

                    # ---- 5) Send result back to client ----
                    await ws.send_json({
                        "type": "voice_result",
                        "transcript": transcript_text,
                        "answer": answer_text,
                        "audio_base64": audio_b64,
                        "audio_mime": "audio/mpeg",
                    })

                else:
                    await ws.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {mtype}",
                    })

    except WebSocketDisconnect:
        print(f"[VOICE][{session_id}] Client disconnected")
    except Exception as e:
        print(f"[VOICE][{session_id}] Unexpected error: {e}")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


if __name__ == "__main__":
    # run: python server.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
