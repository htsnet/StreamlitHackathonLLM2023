import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

schema = {"text": str}
doc_array = Chroma(schema=schema)

st.set_page_config(page_title="LangChain: Chat with Documents", page_icon="🦜")
st.title("🦜 LangChain: Chat with Documents")

text_to_test = """
Hello guys. Welcome to the Eelspace YouTube Channel. Please don't forget to like this video and subscribe to this channel as well. Thank you very much for your support. Due to popular demand, I have recently published a course on a new course on Udemy which will teach you how to build a medical booking application with chat and video call functionalities. Now, this particular course is going to teach you way more than you've ever learnt on these YouTube channel and I definitely recommend you getting the course today.

Now, let me take you through the features of the application in the course. Let's go. Okay, guys, let me walk you through the application that has just been built in the course, right? So first of all, I want to just walk you through the UI UX design that we used in the course. So this is the UIUX design for the course, right? So we have all of these screens, okay?

And of course, full credit goes to the UI UX designer who made these particular designs available for free on the Figma community. Full credit to him. All right, so then these are the screens that you will be able to build in the application, all right? So if you check the description under this video, you're going to see the link to the UI UX design. And then also you will have all of these resources for yourself when you get to purchase the course in Udemy.

Now, let me walk you through the application itself. So everything was built inside Flutterflow, of course, without writing a single line of code, right? So we have all of these screens inside Flutterflow, all of this inside Flutterflow where the user can actually get to interact with the application in different scenarios. So we have the screen where the patients gets to find doctors and then book appointment with the doctors.

And then we also have the admin panel, a basic admin panel which is actually still under construction and of course will be completed in the course. Then we also have the doctors page. All right, so doctors that have been able to get appointments like they've been booked, right? They get to see the appointments that they have and then they can actually go ahead and join the call session. All right, so let's go ahead and then go through the real application itself.

So this is the application, and what I can just quickly do here is I'll run an instant reload so we can just start everything afresh. And then I've also opened the application here, the web published link on one browser, Brave browser, and I've also opened it on this other browser here. Right? So we are going to test all the different users that are in the application, right?

So now we have this, I'm going to go ahead and log in the first kind of user here who is the patient. So I'm just going to log in Jenny. Okay, so let's log in. Jenny, what is happening? So Jenny, jenny@gmail.com and I'm just going to log in the credentials, then you can just go through the application and see it for yourself. Okay. So now this is it, this is the profile for Jenny.

Jenny can actually go ahead and then change her profile using this feature here. Can actually click to upload photo from gallery. All right, so we can actually go ahead and then run all of those updates here and then of course upload the picture and then change this. Then Jenny as a patient can also see the list of doctors, right? So we only have just one doctor right now, which is myself.

So you can actually go ahead and click to see all the doctors and then even see the distance away. Okay. And then you can actually go ahead and click on this to look at the doctor's profile. All right, and then pick a particular date. Maybe you want to have an appointment with doctor on this date or maybe you want a fire date. And then you can pick the time that you want to meet with the doctor and then go ahead and then just book an appointment.

Now booking appointment requires you to make some form of payment. All right, so here I'm going to go ahead and get the test cards from stripe so that I can make payment. So the application is connected to stripe, so users can actually go ahead and make payment with stripe. So you learn how to connect your application to stripe and receive payment. Okay, so now I'm pasting the test card details. I'm adding the required details as well.

Then now I'm going to go ahead and make payment and then after making payments I'm going to see the payment success page, right? So booking successful, your appointment booking is completed. So I can go ahead and click on done. And then now I get to see my other ID. All right, get to see the booking time, like the time I'm supposed to have the meeting with the doctor. So this is it, this is the patient's booking details.

So he's supposed to have a session with the doctor on September 6 and it's going to be by 08:00 a.m., the doctor's name is David and then this is the picture of the doctor and then she can actually go ahead and click here to join the meeting. So I did it in two parts. One part is they can go ahead and then join the meeting on Google link.

Google meet, all right, then eventually I'm going to just go ahead and integrate it such that they can have the video call session on the application itself. Do you understand? Now this is for the part where the user gets to see their bookings, their upcoming bookings. You can click here to see past bookings. So Jenny doesn't have any past bookings. So to actually have a past booking. That means a particular booking has been completed.

So to complete a booking, jenny just has to go ahead and click on this icon here to complete the booking. So there's going to be a prompt here that says, do you want to mark this meeting as completed? I can say no. All right, like this one now. And I can say, do you want to mark this meeting as completed? I can say yes. Now when I mark it as completed, the meeting is out of there. It's not in my past.

All right, so this is the past meeting. Of course, you can also go ahead and click on the chats page to actually send a message to the doctor. All right, so you can click here and then send messages. This is where Jenny and David Ackley had a conversation. Okay, so message chatting feature fully executed. Now we also have okay, this is where Jenny sees the booking details, right? And then chats with the different doctors available.

And also doctors can chat with Jenny as well. Okay, so this is the first part, the user the patient's account. Now let's go ahead and look at the other part. Now let me just log in into the admin panel. The admin panel. The admin panel allows the admin to just see some basic details about the application. All right. But this is going to be improved with time, of course.

So I go ahead and click on this to log in into the admin panel. And then in the admin panel, I get to see the number of users. So if I click here, I get to see the number of users here. I can even set their roles as patients or set them as doctors. Okay, so then I can actually go ahead and then click here to see the doctors that are available.

So it's only one doctor that is available because if you come here, you see that this one says a set as patient. That means this person is a doctor. And then this one says set as doctor. That means this person is a patient. Okay, so that's like that part. And then also the doctors. Okay, so I have a doctor here which is Mr. David. All right, mr. David the rock. And here we can just put in all the information and login.

Now with this particular part. Now the doctor can see their appointment schedules and see the patient that they are supposed to have a session with. So you can see here join noun chats page. They can also click to chat with the patient. So this is where the doctor chats with the patient. All right. And then they can also check their past booking so this doctor doesn't have any past appointments that they've actually completed. Okay, so this is how it works.

And this is how the application works fully. And of course in this application, in this course, I've also made references on different things. How to do web publishing, how to go ahead and then view your code, push your code to GitHub repository, right? So for instance, this is the code in GitHub repository. I pushed it directly from Flutterflow. So in this course, you get to learn how to do that as well.

You learn how to download your code, download the APK file, install it in your Android device, and then see how it works there. Okay? And then also in the course, you're going to learn how to do web publishing, which is actually quite easy anyways. So there are lots and lots of things you're going to learn even more complex backend functionality from just going through this course. So I highly recommend that you go through this course.

Check the link in the description under this video for the link to get the course on Udemy. It's very, very affordable. Udemy is actually making the course super affordable for everyone to get it all right. So just go ahead and then check it out and then give me as much feedback as possible. I would like to get to hear your comments and your response. Don't forget to like this video as well. And of course, subscribe to this YouTube channel. Thank you very much.

And of course I will always keep on providing you with free tutorials, but this is one way of me also like generating revenue from my skill. And so, of course, I super, super much need your support. Go ahead and then check the link and go ahead and enroll in Udemy to get the course. Okay, guys, happy learning and have fun. See you in the next tutorial. Bye for now. Bye.

"""

@st.cache_resource(ttl="1h")
def configure_retriever():
    global text_to_test
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_text(text_to_test)

    # Create embeddings and store in vectordb
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = DocArrayInMemorySearch(splits, embeddings)

    # Define retriever
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

    return retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

retriever = configure_retriever()

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Setup LLM and QA chain
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0, streaming=True
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True
)

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])