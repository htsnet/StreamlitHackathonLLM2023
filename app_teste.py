import streamlit as st
import pandas as pd
from pytube import YouTube
import requests
import time
import assemblyai as aai
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
import plotly.express as px
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

# auth_key from secrets
auth_key = st.secrets['auth_key']

# global variables
audio_location = ''
audio_url = ''
transcription = ''
process_status = ''
link = ''
# to control link already processes
link_new = ''

# youtube-dl options
ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': './%(id)s.%(ext)s',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
}
CHUNK_SIZE = 5242880

# endpoints
upload_endpoint = 'https://api.assemblyai.com/v2/upload'

headers = {
    "authorization": auth_key,
    "content-type": "application/json"    
}

st.set_page_config(page_title='LLM with Streamlit', 
                   page_icon='👀', layout='centered', initial_sidebar_state='expanded' )

# to hide streamlit menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
# pass javascript to hide streamlit menu
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# @st.cache_data(ttl=600)
def download_audio(link):
    global audio_location
    _id = link.strip()
    
    def get_vid(_id):
        # create object YouTube
        yt = YouTube(_id)
        audio_stream = yt.streams.filter(only_audio=True).first()
        # print(audio_stream)
        return audio_stream
        
    # download the audio of the Youtube video locally
    audio_stream = get_vid(_id)
    download_path = './'
    audio_location = audio_stream.download(output_path=download_path)   
    # print('Saved audio to', audio_location)
    
def read_file(filename):
    with open(filename, 'rb') as _file:
        while True:
            data = _file.read(CHUNK_SIZE)
            if not data:
                break
            yield data
            
def upload_audio():
    global audio_location
    global audio_url
    upload_response = requests.post(
        upload_endpoint, 
        headers=headers, 
        data=read_file(audio_location)
    )
    
    audio_url = upload_response.json()['upload_url']


def gauge_chart(value, max_value, label):
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Define angules
    start_angle = 0
    end_angle_red = 180
    end_angle_green = 180 - (value / max_value) * 180 # reverte start point
    
    # arc widgth and radius
    arc_width = 0.2  # width of the arc
    arc_radius = 0.4  # Radius of the arc

    # Arc green
    arc_green = Arc((0.5, 0.5), arc_radius * 2, arc_radius * 2, angle=0, theta1=start_angle, theta2=end_angle_red, color='green', lw=40)
    ax.add_patch(arc_green)

    # Arc red
    arc_red = Arc((0.5, 0.5), arc_radius * 2, arc_radius * 2, angle=0, theta1=start_angle, theta2=end_angle_green, color='red', lw=40)
    ax.add_patch(arc_red)

    # aditional settings
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # explain text
    ax.text(0.5, 0.6, "{:.1f}%".format(round(value, 1)), ha='center', va='center', fontsize=20)
    ax.text(0.5, 0.5, label, ha='center', va='center', fontsize=16)
    ax.text(0.5, 0.25, "Global Confidence", ha='center', va='center', fontsize=26, color='black')
    ax.text(0.5, 0.1, "Greater green bar is better", ha='center', va='center', fontsize=18, color='green')
    return fig


def main():
    global audio_location
    global audio_url
    global process_status
    global transcription
    global link
    global link_new
    
    with st.sidebar:
        # st.image('logo-250-transparente.png')
        st.header('Information')
        st.write("""
                This project was created with the goal of participating in the 'Streamlit LLM Hackathon 2023'. 
                \nThis site use **AssemblyAI** service to transcribe audio from Youtube videos.
                \nAt this point, the video must be in English.
                """)
        
        st.header('About')
        st.write('Details about this project can be found in https://github.com/htsnet/StreamlitHackathonLLM2023')
        
    # título
    title = f'Audio transcription and analysis with LLM'
    st.title(title)
    
    subTitle = f'Using a Youtube video link, this site will transcribe the audio and show relevant information.'
    st.subheader(subTitle)
    
    # information tabs  
    st.markdown('<style>[id^="tabs-bui3-tab-"] > div > p{font-size:20px;}</style>', unsafe_allow_html=True)
    # emoji list https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
    tab1, tab2, tab3, tab4 = st.tabs(['📹:red[ **Video Process**]', '📖:red[ **Transcription**]', '📄:red[ **Sumary**]', '🏷️:red[ **Categories**]'])

    with tab1:
        st.subheader('Start here!')
        
        # link
        link = st.text_input('Paste your Youtube video link and press Enter', value=link)
        
        # download stopwords
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))        
        
        if link != '':
            if link_new != link:
                link_new = link
                time_start = time.time()
                aai.settings.api_key = auth_key
                
                col1, col2, col3 = st.columns(3)
                # using col1 to reduce width of video
                with col1:
                    st.video(link)


                    process_status = 'done'
                    time_stop = time.time()
                    
                
                with col2:
                    time_total = time_stop - time_start
                    st.write('🕔 Processed in',  "{:.1f}".format(round(time_total, 1)), 'seconds!')
                    
                st.markdown('See the tabs above for information about the audio!')
                if st.button("Reset", type="primary"):
                    audio_location = ''
                    audio_url = ''
                    process_status = ''
                    transcription = ''
                    link = ''
                    st.experimental_rerun()

if __name__ == '__main__':
	main()   