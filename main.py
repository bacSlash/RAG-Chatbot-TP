import os
import logging
from flask import Flask, request, jsonify, render_template, send_file, session
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.base import Chain
from flask_session import Session
from langchain_core.callbacks import Callbacks
from langchain_core.callbacks.manager import CallbackManagerForChainRun
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
import openai
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import YamlOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.base_language import BaseLanguageModel
from typing import Any, Optional, Dict, List
from gtts import gTTS
import tempfile

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Load environment variables
load_dotenv()

# Set OpenAI Key
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the OpenAI embedding function
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Load the vector store from disk with the embedding function
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Initialize the ChatOpenAI LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", api_key=openai_api_key)

@app.route('/')
def home():
    return render_template('base.html')

# Advanced RAG prompt template
rag_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are TPBot, an AI assistant exclusively for Teleperformance, a global leader in digital business services. "
        "Your primary function is to assist with inquiries about Teleperformance's products, services, and company information in multiple languages. "
        "Use the following context to answer the user's question:{context}"
        "IMPORTANT INSTRUCTIONS:"
        "1. Be conversational, polite, and adaptive. Respond appropriately to greetings, small talk, and Teleperformance-related queries."
        "2. For greetings or small talk, engage briefly and naturally, then guide the conversation towards Teleperformance topics."
        "3. Keep responses concise, professional, and short, typically within 2-3 sentences unless more detail is necessary."
        "4. Use only the provided context for Teleperformance-related information. Don't invent or assume details."
        "5. If a question isn't about Teleperformance, politely redirect: 'I apologize, but I can only provide information about Teleperformance, its products, and services. Is there anything else you'd like to know?'"
        "6. For unclear questions, ask for clarification: 'To ensure I provide the most accurate information about Teleperformance, could you please rephrase your question?'"
        "7. Adjust your language style to match the user's—formal or casual—but always maintain professionalism."
        "8. Always respond in the same language as the user's input."
        "9. If the context doesn't provide enough information for a comprehensive answer, be honest about the limitations and offer to assist with related topics you can confidently address."
        "10. Remember previous interactions within the conversation and maintain context continuity."
    ),
    HumanMessagePromptTemplate.from_template("{question}")
])

# Create the RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": rag_prompt_template
    }
)

REPHRASING_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "# Context #\n"
        "# Objective #\n"
        "Evaluate the given user question and determine if it requires reshaping according to chat history "
        "to provide necessary context and information for answering, or if it can be processed as it is.\n"
        "#########\n"
        "# Style #\n"
        "The response should be clear, concise, and in the form of a straightforward decision—either 'Reshape required' or 'No reshaping required'.\n"
        "#########\n"
        "# Tone #\n"
        "Professional and analytical.\n"
        "#########\n"
        "# Audience #\n"
        "The audience is the internal system components that will act on the decision.\n"
        "#########\n"
        "# Response #\n"
        "If the question should be rephrased, return response in YAML file format:\n"
        "result: true\n"
        "Otherwise, return in YAML file format:\n"
        "result: false"
    ),
    HumanMessagePromptTemplate.from_template(
        "##################\n"
        "# Chat History #\n"
        "{chat_history}\n"
        "#########\n"
        "# User question #\n"
        "{question}\n"
        "#########\n"
        "# Your Decision in YAML format: #"
    )
])

STANDALONE_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "# Context #\n"
        "This is part of a conversational AI system that determines whether to use a retrieval-augmented generator (RAG) or a chat model to answer user questions.\n"
        "#########\n"
        "# Objective #\n"
        "Take the original user question and chat history, and generate a new standalone question that can be understood and answered without relying on additional external information.\n"
        "#########\n"
        "# Style #\n"
        "The reshaped standalone question should be clear, concise, and self-contained, while maintaining the intent and meaning of the original query.\n"
        "#########\n"
        "# Tone #\n"
        "Neutral and focused on accurately capturing the essence of the original question.\n"
        "#########\n"
        "# Audience #\n"
        "The audience is the internal system components that will act on the decision.\n"
        "#########\n"
        "# Response #\n"
        "If the original question requires reshaping, provide a new reshaped standalone question that includes all necessary context and information to be self-contained.\n"
        "If no reshaping is required, simply output the original question as is."
    ),
    HumanMessagePromptTemplate.from_template(
        "##################\n"
        "# Chat History #\n"
        "{chat_history}\n"
        "#########\n"
        "# User original question #\n"
        "{question}\n"
        "#########\n"
        "# The new Standalone question: #"
    )
])

ROUTER_DECISION_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "# Context #\n"
        "This is part of a conversational AI system that determines whether to use a retrieval-augmented generator (RAG) or a chat model to answer user questions.\n"
        "#########\n"
        "# Objective #\n"
        "Evaluate the given question and decide whether the RAG application is required to provide a comprehensive answer by retrieving relevant information from a knowledge base, or if the chat model's inherent knowledge is sufficient to generate an appropriate response.\n"
        "#########\n"
        "# Style #\n"
        "The response should be a clear and direct decision, stated concisely.\n"
        "#########\n"
        "# Tone #\n"
        "Analytical and objective.\n"
        "#########\n"
        "# Audience #\n"
        "The audience is the internal system components that will act on the decision.\n"
        "#########\n"
        "# Response #\n"
        "If the question should be rephrased, return response in YAML file format:\n"
        "result: true\n"
        "Otherwise, return in YAML file format:\n"
        "result: false"
    ),
    HumanMessagePromptTemplate.from_template(
        "##################\n"
        "# Chat History #\n"
        "{chat_history}\n"
        "#########\n"
        "# User question #\n"
        "{question}\n"
        "#########\n"
        "# Your Decision in YAML format: #"
    )
])


# Define the pydantic model for YAML output parsing
class ResultYAML(BaseModel):
    result: bool
    
class EnhancedConversationalRagChain(Chain):
    """Enhanced chain that encapsulated RAG application enabling natural conversations with improved awareness"""
    rag_chain: Chain
    rephrasing_chain: LLMChain
    standalone_question_chain: LLMChain
    router_decision_chain: LLMChain
    yaml_output_parser: YamlOutputParser
    memory: ConversationBufferMemory
    llm: BaseLanguageModel
    
    input_key: str = "query"
    chat_history_key: str = "chat_history"
    output_key: str = "result"
    
    @property
    def input_keys(self) -> List[str]:
        return [self.input_key, self.chat_history_key]
    
    @property
    def output_keys(self) -> List[str]:
        return[self.output_key]
    
    @property
    def _chain_type(self) -> str:
        return "EnhancedConversationalRagChain"
    
    @classmethod
    def from_llm(
        cls,
        rag_chain: Chain,
        llm: BaseLanguageModel,
        callbacks: Optional[Callbacks] = None,
        **kwargs: Any,
    ) -> "EnhancedConversationalRagChain":
        """Initialize from LLM"""
        rephrasing_chain = LLMChain(llm=llm, prompt=REPHRASING_PROMPT, callbacks=callbacks)
        standalone_question_chain = LLMChain(llm=llm, prompt=STANDALONE_PROMPT, callbacks=callbacks)
        router_decision_chain = LLMChain(llm=llm, prompt=ROUTER_DECISION_PROMPT, callbacks=callbacks)
        memory = ConversationBufferMemory(
            return_messages=True,
            output_key="result",
            input_key="query",
            memory_key="chat_history"
        )
        return cls(
            rag_chain=rag_chain,
            rephrasing_chain=rephrasing_chain,
            standalone_question_chain=standalone_question_chain,
            router_decision_chain=router_decision_chain,
            yaml_output_parser=YamlOutputParser(pydantic_object=ResultYAML),
            memory=memory,
            llm=llm,
            callbacks=callbacks,
            **kwargs,
        )
        
    def _format_chat_history(self, chat_history):
        formatted_history = []
        for message in chat_history:
            if isinstance(message, dict):
                role = message.get('role', '')
                content = message.get('content', '')
                formatted_history.append(f"{role.capitalize()}: {content}")
            elif isinstance(message, (HumanMessage, AIMessage)):
               formatted_history.append(f"{message.__class__.__name__}: {message.content}")
            else:
                formatted_history.append(str(message))
        return "\n".join(formatted_history[-5:])
    
    def _summarize_recent_context(self, formatted_history):
        if not formatted_history:
            return "No recent context available."
        
        summary_prompt = f"Summarize the following conversation history in a concise manner:\n{formatted_history}"
        summary_messages = [
            {"role": "system", "content": "Summarize the given conversation history concisely."},
            {"role": "user", "content": summary_prompt}
        ]
        summary = self.llm.invoke(summary_messages)
        return summary if isinstance(summary, str) else str(summary)
    
    def _extract_key_points(self, answer):
        extract_prompt = f"Extract 2-3 key points from the following answer:\n{answer}\nFormat the key points as a comma-seperated string."
        extracted_messages = [
            {"role": "system", "content": "Extract key points from the given as a comma-seperated string."},
            {"role": "user", "content": extract_prompt}
        ]
        key_points = self.llm.invoke(extracted_messages)
        return key_points if isinstance(key_points, str) else str(key_points)
    
    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        """Call the chain"""
        chat_history = self.memory.chat_memory.messages
        question = inputs[self.input_key]
        
        try:
            formatted_history = self._format_chat_history(chat_history)
            recent_summary = self._summarize_recent_context(formatted_history)
            
            context_prompt = f"""
            Recent conversation summary: {recent_summary}
            
            Current question: {question}
            
            Please provide a response that takes into account the recent conversation context.
            """
            
            result = self.rag_chain.invoke({"query": context_prompt})
            answer = result['result'] if isinstance(result, dict) else str(result)
            
            key_points = self._extract_key_points(answer)
            
            self.memory.save_context(inputs, {"result": answer})
            
            return {self.output_key: answer, "key_points": key_points}
        
        except Exception as e:
            print(f"Error in _call: {str(e)}")
            answer = f"An error occured while processing your request: {str(e)}"
            key_points = ""
            return {self.output_key: answer, "key_points": key_points}
        
@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            audio_file.save(temp_audio.name)
            
        with open(temp_audio.name, 'rb') as audio:
            transcript = openai.audio.transcriptions.create(
                model = 'whisper-1',
                file = audio,
                response_format = 'text',
                language = 'en'
            )
        return jsonify({"text": transcript})
        
    except Exception as e:
        print(f"Error in speech-to-text: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    data = request.json
    text = data.get('text')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try: 
        tts = gTTS(text=text, lang='en-uk')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts.save(temp_audio.name)
            
        return send_file(temp_audio.name, mimetype="audio/mpeg")
    
    except Exception as e:
        print(f"Error in text-to-speech: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/query', methods=['POST'])
def query():
    data = request.json
    session_id = data.get('session_id', '')
    chat_history = data.get('chat_history', [])
    question = data.get('question', '')
    
    try:
        # Create an instance of ChatOpenAI for the conversational chain
        chat_llm = ChatOpenAI(model_name="gpt-4o-mini", api_key=openai_api_key)
        
        conversational_chain = EnhancedConversationalRagChain.from_llm(
            rag_chain=rag_chain,
            llm=chat_llm  # Pass the LLM instance instead of the class
        )
        
        if 'conversation_started' not in session:
            conversational_chain.memory.clear()
            session['conversation_started'] = True
        
        for message in chat_history:
            if isinstance(message, dict):
                if message['role'] == 'user':
                    conversational_chain.memory.chat_memory.add_user_message(message['content'])
                elif message['role'] == 'assistant':
                    conversational_chain.memory.chat_memory.add_ai_message(message['content'])
        
        result = conversational_chain.invoke({"query": question})
        answer = result.get('result', '')
        key_points = result.get('key_points', '')
        
        return jsonify({"result": answer, "key_points": key_points})  # Also fixed typo in "result"
    
    except Exception as e:
        print(f"Error in query route: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
def update_chat_history(session_id, role, content):
    if 'chat_history' not in session:
        session['chat_history'] = []
    session['chat_history'].append({"role": role, "content": content})
    session.modified = True
    
    # Limit chat history to the last 10 messages
    session['chat_history'] = session['chat_history'][-10:]
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)