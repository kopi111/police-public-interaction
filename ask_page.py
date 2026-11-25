"""A chat page for asking the assistant questions. Run it with: .venv/bin/python ask_page.py"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from stop_question_assistant import AssistantUnavailable, build_assistant

# Only this computer can open the page, because of the 127.0.0.1 address.
HOST = "127.0.0.1"
PORT = 5385
LONGEST_QUESTION = 500
LONGEST_REQUEST_BYTES = 4096

# The page. Replies are added with textContent, so text can never run as code in the browser.
CHAT_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Police Stops Assistant</title>
<link rel="icon" href="data:,"><style>
:root { --surface: #fcfcfb; --panel: #ffffff; --ink: #0b0b0b; --muted: #52514e; --line: #d9d8d2; --blue: #2a78d6; --tint: #e8f1fc; --error: #b3261e; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--ink); font: 16px/1.5 system-ui, sans-serif; }
main { max-width: 720px; margin: 0 auto; padding: 24px 16px 140px; }
h1 { font-size: 24px; margin: 0 0 4px; }
.note { color: var(--muted); font-size: 14px; margin: 0 0 16px; }
.message { padding: 12px 16px; border-radius: 8px; margin-bottom: 12px; white-space: pre-wrap; overflow-wrap: anywhere; }
.you { background: var(--tint); margin-left: 15%; }
.assistant { background: var(--panel); border: 1px solid var(--line); margin-right: 5%; }
.failed { color: var(--error); font-weight: 600; }
details { margin-top: 8px; color: var(--muted); font-size: 14px; white-space: normal; }
.examples { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
.examples button { background: var(--panel); color: var(--ink); border: 1px solid var(--line); font-weight: 400; font-size: 14px; }
form { position: fixed; bottom: 0; left: 0; right: 0; background: var(--surface); border-top: 1px solid var(--line); padding: 12px 16px; }
.row { max-width: 688px; margin: 0 auto; display: flex; gap: 8px; }
input { flex: 1; min-width: 0; padding: 12px; font: inherit; border: 1px solid var(--line); border-radius: 6px; }
button { padding: 12px 18px; font: inherit; font-weight: 600; color: #fff; background: var(--blue); border: 0; border-radius: 6px; cursor: pointer; }
button:disabled { opacity: .6; cursor: wait; }
</style></head><body><main>
<h1>Ask about Seattle police stops</h1>
<p class="note">Learning project. Answers come from recorded Seattle Terry stops, 2015 to 3 November 2026. Each answer takes about 30 seconds.</p>
<div class="examples" id="examples"></div><div id="chat" aria-live="polite"></div></main>
<form id="form"><div class="row"><input id="question" maxlength="500" placeholder="Ask a question" aria-label="Your question" autocomplete="off">
<button id="send">Ask</button></div></form>
<script>
const EXAMPLES = ["How many stops were there in 2024?", "What share of shoplifting calls end in arrest?", "Which precinct had the most arrests last year?", "Arrest rate by perceived race in 2025", "Chance of arrest for a felony warrant pickup in beat K3?", "How accurate are you?"];
const chat = document.getElementById("chat"), input = document.getElementById("question"), send = document.getElementById("send");
function addMessage(text, kind) {
  const box = document.createElement("div");
  box.className = "message " + kind;
  box.textContent = text;
  chat.appendChild(box);
  box.scrollIntoView({ block: "nearest" });
  return box;
}
function addWorking(box, working) {
  if (!working.length) return;
  const details = document.createElement("details"), summary = document.createElement("summary");
  summary.textContent = "How I worked this out";
  details.appendChild(summary);
  working.forEach(line => { const item = document.createElement("div"); item.textContent = line; details.appendChild(item); });
  box.appendChild(details);
}
async function ask(question) {
  addMessage(question, "you");
  const waiting = addMessage("Working it out...", "assistant");
  send.disabled = true;
  try {
    const response = await fetch("/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }) });
    const answer = await response.json();
    if (!response.ok) throw new Error(answer.error);
    waiting.textContent = answer.text;
    addWorking(waiting, answer.working);
  } catch (error) {
    waiting.textContent = error.message || "Something went wrong. Try again.";
    waiting.classList.add("failed");
  }
  send.disabled = false;
  input.focus();
}
document.getElementById("form").addEventListener("submit", event => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question || send.disabled) return;
  input.value = "";
  ask(question);
});
EXAMPLES.forEach(example => {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = example;
  button.onclick = () => { if (!send.disabled) ask(example); };
  document.getElementById("examples").appendChild(button);
});
input.focus();
</script></body></html>"""

# Load the data once. The lock makes questions take turns, because the laptop can only run one at a time.
assistant = build_assistant()
assistant_lock = threading.Lock()


class ChatPageHandler(BaseHTTPRequestHandler):
    """Serves the page and answers questions."""

    def do_GET(self):
        """Send the chat page."""
        if self.path != "/":
            self.send_error(404)
            return
        self.send_body(200, CHAT_PAGE.encode(), "text/html; charset=utf-8")

    def do_POST(self):
        """Answer one question, sending a short message if it fails."""
        if self.path != "/ask" or not self.is_from_this_page():
            self.send_error(403)
            return
        try:
            self.send_json(200, answer_question(self.read_question()))
        except ValueError as problem:
            self.send_json(400, {"error": str(problem)})
        except AssistantUnavailable as problem:
            self.log_error("%s", problem)
            self.send_json(503, {"error": "The language model isn't running. Start Ollama and try again."})
        except Exception as problem:
            self.log_error("Could not answer: %r", problem)
            self.send_json(500, {"error": "Something went wrong. Try asking another way."})

    def is_from_this_page(self):
        """Block other websites from sending questions here through the browser."""
        return self.headers.get("Content-Type") == "application/json" and self.headers.get("Host") == f"{HOST}:{PORT}"

    def read_question(self):
        """Read and check the question, raising an error that names the problem."""
        length = int(self.headers.get("Content-Length", 0))
        if length > LONGEST_REQUEST_BYTES:
            raise ValueError("Your question is too long.")
        question = self.read_question_text(length)
        if not question:
            raise ValueError("Type a question first.")
        if len(question) > LONGEST_QUESTION:
            raise ValueError(f"Your question is too long. Keep it under {LONGEST_QUESTION} characters.")
        return question

    def read_question_text(self, length):
        """Pull the question out of the request, with a short message if the request is malformed."""
        try:
            return str(json.loads(self.rfile.read(length)).get("question", "")).strip()
        except (json.JSONDecodeError, AttributeError) as problem:
            self.log_error("Malformed request: %r", problem)
            raise ValueError("Type a question first.") from problem

    def send_json(self, status, content):
        """Send data back to the page."""
        self.send_body(status, json.dumps(content).encode(), "application/json")

    def send_body(self, status, body, content_type):
        """Send a finished response."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def answer_question(question):
    """Ask the assistant, one question at a time."""
    with assistant_lock:
        answer = assistant.ask(question)
    return {"text": answer.text, "working": answer.working}


if __name__ == "__main__":
    print(f"Open http://{HOST}:{PORT} in your browser. Press Ctrl+C to stop.")
    ThreadingHTTPServer((HOST, PORT), ChatPageHandler).serve_forever()
