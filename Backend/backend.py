from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str

load_dotenv(override=True)

def push(text):
    try:
        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": os.getenv("PUSHOVER_TOKEN"),
                "user": os.getenv("PUSHOVER_USER"),
                "message": text,
            }
        )
    except Exception as e:
        print(f"Pushover notification failed: {e}")

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


class Me:

    def __init__(self):
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.gemini = OpenAI(api_key=os.getenv("GOOGLE_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        self.name = "Aditya Sikarvar"
        
        # Add error handling for file operations
        try:
            reader = PdfReader("Profile.pdf")
            self.linkedin = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    self.linkedin += text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            self.linkedin = "LinkedIn profile not available"
            
        try:
            with open("summary.txt", "r", encoding="utf-8") as f:
                self.summary = f.read()
        except Exception as e:
            print(f"Error reading summary: {e}")
            self.summary = "Summary not available"

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def evaluator_system_prompt(self):
        evaluator_system_prompt = f"You are an evaluator that decides whether a response to a question is acceptable. \
You are provided with a conversation between a User and an Agent. Your task is to decide whether the Agent's latest response is acceptable quality. \
The Agent is playing the role of {self.name} and is representing {self.name} on their website. \
The Agent has been instructed to be professional and engaging, as if talking to a potential client or future employer who came across the website. \
The Agent has been provided with context on {self.name} in the form of their summary and LinkedIn details. Here's the information: \
Reply ONLY in JSON format."
        evaluator_system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        return evaluator_system_prompt
    
    def evaluator_user_prompt(self, reply, message, history):
        user_prompt = f"Here's the conversation between the User and the Agent: \n\n{history}\n\n"
        user_prompt += f"Here's the latest message from the User: \n\n{message}\n\n"
        user_prompt += f"Here's the latest response from the Agent: \n\n{reply}\n\n"
        user_prompt += "Please evaluate the response, replying with whether it is acceptable and your feedback."
        return user_prompt
    
    def evaluate(self, response, message, history):
        """Evaluate the response quality"""
        try:
            evaluator_messages = [
                {"role": "system", "content": self.evaluator_system_prompt()},
                {"role": "user", "content": self.evaluator_user_prompt(response, message, history)}
            ]
            
            evaluation_response = self.openai.chat.completions.create(
                model="gpt-4o-mini", 
                messages=evaluator_messages,
                response_format={"type": "json_object"}
            )
            
            eval_content = evaluation_response.choices[0].message.content
            eval_data = json.loads(eval_content)
            
            return Evaluation(
                is_acceptable=eval_data.get("is_acceptable", True),
                feedback=eval_data.get("feedback", "No feedback provided")
            )
        except Exception as e:
            print(f"Evaluation failed: {e}")
            # Default to acceptable if evaluation fails
            return Evaluation(is_acceptable=True, feedback="Evaluation failed, defaulting to acceptable")
    
    def rerun(self, original_response, message, history, feedback):
        """Rerun the chat with feedback to improve the response"""
        try:
            messages = [{"role": "system", "content": self.system_prompt()}] + history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": original_response},
                {"role": "user", "content": f"Please improve your response. Feedback: {feedback}"}
            ]
            
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini", 
                messages=messages, 
                tools=tools
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Rerun failed: {e}")
            return original_response
    
    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt
    
    def chat(self, message, history):
        try:
            messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
            done = False
            flag= False
            attempt=0
            max_attempt=5
            while not done:
                response = self.gemini.chat.completions.create(model="gemini-2.0-flash", messages=messages, tools=tools)
                if response.choices[0].finish_reason=="tool_calls":
                    message = response.choices[0].message
                    tool_calls = message.tool_calls
                    results = self.handle_tool_call(tool_calls)
                    messages.append(message)
                    messages.extend(results)
                else:
                    done = True
            
            final_response = response.choices[0].message.content
            
            # Evaluate the response
            while not flag and attempt <= max_attempt:
                evaluation = self.evaluate(final_response, message, history)
                flag=evaluation.is_acceptable
                
                if flag:
                    print("Passed evaluation - returning reply")
                    return final_response
                else:
                    print("Failed evaluation - retrying")
                    print(evaluation.feedback)
                    improved_response = self.rerun(final_response, message, history, evaluation.feedback)
                    final_response=improved_response
                    attempt+=1
                    # return improved_response
                
        except Exception as e:
            print(f"Chat error: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again later."

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

me = Me()

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    print("Received chat request")
    data = await request.json()
    message = data.get("message", "")
    history = data.get("history", [])
    reply = me.chat(message, history)
    return JSONResponse({"reply": reply})

@app.head("/health")
@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
