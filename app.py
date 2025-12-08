import os
import dotenv
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# ==========================
#   ENV + BASE SETUP
# ==========================

dotenv.load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

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

app = FastAPI(title="SQL Chat Backend")

# CORS so you can call from a frontend (React, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in prod, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "ok", "message": "SQL chat backend running"}


@app.websocket("/ws/sql-chat")
async def sql_chat(ws: WebSocket):
    """
    WebSocket endpoint for text-only chat with the SQL agent.

    Protocol (simple):
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


if __name__ == "__main__":
    # run: python server.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
