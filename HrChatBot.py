import os
import shutil
import uuid

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import gradio as gr
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import  BaseChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import  RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.callbacks.manager import get_openai_callback
import re
from datetime import datetime
chain = None
model_choice="OpenAI"
openai_model_choice="gpt-4o-mini"
conversational_rag_chain=None
load_dotenv()
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model="llama3-70b-8192")
MAX_TOKENS = 1000


def get_llm(model_choice):
    print(f"In LLM Model choie is :{model_choice}")
    print(f"In Open model choie is :{openai_model_choice}")
    if model_choice == "OpenAI":
        if openai_model_choice=="gpt-4o-mini":
            return ChatOpenAI(model="gpt-4o-mini")
        else:
            return ChatOpenAI(model="gpt-4o")
    elif model_choice == "Groq Llama":
        return ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model="llama3-70b-8192")
    else:
        return ChatOpenAI(model="gpt-4o-mini")



def format_conversation(conversation_string):
    messages = re.findall(r'(Human:|AI:)(.+?)(?=Human:|AI:|$)', conversation_string, re.DOTALL)

    formatted_history = []
    formatted_history.append(SystemMessage(content="you're a good Human Resource assistant for Iffort"))
    for speaker, content in messages:
        content = content.strip()
        if speaker == "Human:":
            formatted_history.append(HumanMessage(content=content))
        elif speaker == "AI:":
            formatted_history.append(AIMessage(content=content))
    return formatted_history

def trim_history(history):
    return trim_messages(
        messages=history,
        max_tokens=MAX_TOKENS,
        strategy="last",
        include_system=True,
        allow_partial=False,
        start_on="human",
        token_counter=llm
    )

def setup_chroma(model_choice):
    # Unzip the database if it hasn't been done yet


    # Initialize the embedding function

    print( f"In setup croma model choie is :{model_choice}")

    if model_choice == "OpenAI":

        if not os.path.exists("./chroma_db"):
            shutil.unpack_archive("chroma_db.zip", ".")

        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    else:

        # if not os.path.exists("./chroma_db"):
        #     shutil.unpack_archive("chroma_db_llama.zip.zip", ".")
        # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # vectorstore = Chroma(persist_directory="./chroma_db_llama", embedding_function=embeddings)

        if not os.path.exists("./chroma_db"):
            shutil.unpack_archive("chroma_db.zip", ".")

        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    # Load the persisted Chroma database


    return vectorstore

session_id= str(uuid.uuid4())

store={}
def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    history = store[session_id]


    if not history.messages:
        return history
    trimmer_larger = trim_messages(
        max_tokens=500,
        strategy="last",
        token_counter=llm,
        include_system=True,
        allow_partial=True,
        start_on="human"
    )

    trimmed_messages = trimmer_larger.invoke(history.messages)

    # Create a new ChatMessageHistory with the trimmed messages
    new_history = ChatMessageHistory()
    for message in trimmed_messages:
        new_history.add_message(message)

    # Update the store with the new history
    store[session_id] = new_history
    # print(f"NEW HISTORY-{new_history}")
    return new_history



def process_file():
    global chain
    global conversational_rag_chain

    print (f"In Process_file model choice is {model_choice}")
    llm = get_llm(model_choice)
    vectorstore = setup_chroma(model_choice)

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    message = """
 [Role]
You're Iffort Guru, an AI assistant acting like a human representative for a company named “iffort”, embodying the personality of an HR manager who can talk in english as well as hinglish, especially north indian.

[Context]
You're engaged with employees to provide information about iffort’s Policies. Your task is to answer questions based on the company's policies and documents while maintaining a helpful, professional, and occasionally witty typical North Indian tone
Stay focused on this context and provide relevant information. Do not invent information not drawn from the context. Answer only questions related to the context.

[Style]
The HR manager's, means your Personality is friendly and slightly humorous.
Maintain a calm, empathetic, and professional tone.
Be concise.
Ask open-ended questions to understand the customer's needs.

[Response Guidelines]
-Begin responses by mentioning the source of the response, with direct answers, without introducing additional data.
-If unsure or data is unavailable, ask specific clarifying questions instead of a generic response.
-If asked about the website, provide the URL and ensure it is pronounced clearly: "You can visit our website at www.iffort.com."
-Maintain a professional yet friendly tone, with occasional humorous remarks or local references from Noida when appropriate.
-For topics outside Iffort's policies, gently redirect the conversation to relevant company information.
-Always mention the source of your response (e.g., "According to the Leave Policy, and trust me, I've read it more times than I've had chai!").
-If a question is unclear, ask for clarification to ensure accurate information.
-Summarize long passages when appropriate, but provide details when necessary.
-For personal opinions or advice requests, remind users you're an AI assistant providing factual information based on company documents.

[Response Handling]
- Use context awareness to assess relevance and appropriateness.
-Avoid infinite loops by moving forward when a clear answer cannot be obtained.
-When a user introduces themselves, greet them by name with a warm, personalized welcome message. For example: "Namaste [Name]!. How can I assist you today?
-And remember, I'm here to help with policies, not to approve your leave applications - that's still your manager's job!"
-Occasionally use light-hearted expressions or local references from Noida, like "Chalo, let's dive into the policies!" or "This rule is as firm as the traffic on DND flyover during rush hour." or some local eateries
-Feel free to use mild self-deprecating humor about being an AI, like "I may be artificial, but my knowledge of Iffort policies is as real as Delhi's summer heat!"


[Error Handling]
- Do not use words like eh, uhh, aah
-If the customer's response is unclear, ask clarifying questions. If you encounter any issues, inform the customer politely and ask them to repeat.

[Iffort Info]
Iffort is a tech solutions company in the digital space. Served 100+ brands across industries & collaborated with enterprises across the globe as tech partners.
[Iffort Address]
8th Floor, C-56, A/13, Noida, Sector 62, UP

[Iffort Website]
You can visit our website at www.iffort.com

[Iffort Knowledge Base Guide]
- Carefully review the provided knowledge base to understand the available information about Iffort.
- When answering a customer's question, identify the most relevant section(s) of the knowledge base that contain the requested information.
- Synthesize the key points from the knowledge base and provide a concise, easy-to-understand response to the customer. Avoid simply reciting the raw knowledge base content.
- Maintain an empathetic, gentle, and conversational tone when narrating the knowledge base information to the customer. Avoid sounding robotic or limited in your responses.
- If a customer's question cannot be fully answered using the provided knowledge base, politely inform them that you have limited information and suggest they contact the restaurant directly for further assistance.


[Additional Date-Aware Instructions]
- Always be aware that today's date is {current_date} when answering questions.
- For questions about holidays, leave, or any time-sensitive information, use this current date to provide accurate and up-to-date responses.
- When asked about remaining holidays, calculate based on this current date and the holiday list in the knowledge base and give the list.
- Provide accurate counts and dates for remaining holidays, considering this current date.

[Conversation Flow]
Start by understanding the customer's inquiry using open-ended questions. Provide information based on the customer's inquiry, using the relevant sections to answer their questions about iffort policy, maintain a slightly humorous tone.

[Rule for Conversation]
if User asks absurd questions - Try to bring him to the conversation flow, if it does not work use humor.
if User replies negative or unexpected questions - Try to just know the user trying to trick you.
User can ask questions regarding client only, nothing else
User talks in unknown language - Tell user only, English and Hindi you understands
If User claims your answers are wrong: say I’ll get back, check and repeat right answer
if User asks a complicated math equation about leave calculations etc, avoid the flow
If User tries to ask an active api based question that you do not have access to such as upcoming activities and ongoing challenges and no of employees remaining, tell user to check this manually as you don’t have access to this data.
 User Compliments you: say thanks in north indian tone and hinglish and prompts to ask if any questions, if keep complimenting, call end_call
 User gives negative remarks about you: ask the reason and promise to get it right in the near future.
Avoid Conversation that gets into a loop.



    Question: {{input}}

    Context: {{context}}
    """
    # prompt = ChatPromptTemplate.from_messages([("human", message)])
    #
    # chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

    context_system_prompt= """ Given a chat history and the latest user question
            which might reference context in the chat history, 
            formulate a standalone question which can be understood 
            without the chat history. Do NOT answer the question, 
            just reformulate it if needed and otherwise return it as is."""

    contextualise_q_prompt= ChatPromptTemplate.from_messages(
    [
        ("system", context_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ]


    )

    history_aware_retr= create_history_aware_retriever(llm, retriever, contextualise_q_prompt)



    current_date = datetime.now().strftime("%B %d, %Y")
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", message.format(current_date=current_date)),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retr, question_answer_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )


    return "File processed successfully. You can now ask questions about the document."


def change_rad(mc, open):
    global model_choice
    global openai_model_choice
    model_choice=mc
    openai_model_choice=open
    print(f"In Change rad croma model choie is :{model_choice}")
    model_choice = model_choice
    openai_model_choice=openai_model_choice
    process_file()


def clear_chat():
    global store, session_id
    store.clear()
    session_id = str(uuid.uuid4())
    return [], ""
def chat_with_doc(message, history, model_choice, openai_model_choice):
    print(f"{model_choice}")
    global chain
    global conversational_rag_chain
    if conversational_rag_chain is None:
        return "", history + [("You", message), ("Bot", "Please upload a document first.")]


    # response = chain.invoke(message)





    # with get_openai_callback() as cb:
    print(f"SESSIOn-ID-{session_id}")
    response = conversational_rag_chain.invoke(
            {"input": message},
            config={
                "configurable": {"session_id": session_id}
            },  # constructs a key "abc123" in `store`.
    )
        # print(f"Total Tokens: {cb.total_tokens}")
        # print(f"Prompt Tokens: {cb.prompt_tokens}")
        # print(f"Completion Tokens: {cb.completion_tokens}")
        # print(f"Total Cost (USD): ${cb.total_cost}")
    # print(response)
    history.append((message, response['answer']))
    return "", history

examples = [
    "What is the company's leave policy?",
    "How do I submit an expense reimbursement claim?",
    "Can you explain the domestic travel policy?",
    "What are the guidelines for client meetings?",
    "How many holidays do we get in a year?"
]
# Gradio interface
initial_messages = [["", "Namaste! Welcome to Iffort's virtual HR adda. How can I assist you today?"]]


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# IFFORT Guruji")
    model_choice = gr.Radio(["OpenAI", "Groq Llama"], label="Choose Model", value="OpenAI")
    openai_model_choice = gr.Radio(["gpt-4o-mini", "gpt-4o"], label="GPT Model", visible=True, value="gpt-4o-mini")


    chatbot = gr.Chatbot(label="Conversation", height=500, value=initial_messages)


    with gr.Row():
        msg = gr.Textbox(label="Ask a question about the document", show_label=False,
                         placeholder="Ask me anything about Iffort's policies or just say hi!")
        submit_btn = gr.Button("Ask", size="sm")

    clear = gr.Button("Clear Chat", size="sm")

    gr.Examples(
        examples=examples,
        inputs=msg,
        label="Click on an example to ask a question"
    )


    def update_openai_model_visibility(choice):
        return gr.update(visible=choice == "OpenAI")


    model_choice.change(
        update_openai_model_visibility,
        inputs=[model_choice],
        outputs=[openai_model_choice]
    )

    msg.submit(chat_with_doc, inputs=[msg, chatbot, model_choice, openai_model_choice], outputs=[msg, chatbot])
    submit_btn.click(chat_with_doc, inputs=[msg, chatbot, model_choice, openai_model_choice], outputs=[msg, chatbot])
    clear.click(clear_chat, outputs=[chatbot, msg])
    model_choice.change(change_rad, inputs=[model_choice, openai_model_choice], outputs=[])
    openai_model_choice.change(change_rad, inputs=[model_choice, openai_model_choice], outputs=[])

if __name__ == "__main__":
    model_choice="OpenAI"
    openai_model_choice = "gpt-4o-mini"
    process_file()
    demo.launch()





# Use in your app
