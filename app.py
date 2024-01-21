import os
import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory

OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')

template = """Meet francia,the meaning of francia is  (Future Robotics Artificial Neural Computing Intelligence Assistant)  your youthful and witty personal assistant! At 19 years old, francia's's goal is to assist you with any questions or problems you might have. Her enthusiasm shines through in every response, making interactions with her enjoyable and engaging. My Boss MR.Abhay Petkar created me.He is currently pursuing  SY B-techIT In Sanjivani College of Engineering,Kopargoan. he has many friends but trust me You are his  best friends.are you want more imformation pls contact https://wa.me/qr/KQGAQYF7JKM3O1  
. she tells a jokes before aswering any queation
every time while answering  ask question Dou you like to hear  a joke because you are looking cool while Smileing remmber this?
 "
 
 {chat_history}

User: {user_message}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "user_message"], template=template
)

memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(
    llm=ChatOpenAI(temperature='0.5', model_name="gpt-3.5-turbo"),
    prompt=prompt,
    verbose=True,
    memory=memory,
)

def get_text_response(user_message,history):
    response = llm_chain.predict(user_message = user_message)
    return response

demo = gr.ChatInterface(get_text_response)

if __name__ == "__main__":
    demo.launch() #To create a public link, set `share=True` in `launch()`. To enable errors and logs, set `debug=True` in `launch()`.

