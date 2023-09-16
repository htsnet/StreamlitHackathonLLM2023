import streamlit as st
from pytube import YouTube
import requests
import pprint
from time import sleep

# auth_key from secrets
auth_key = st.secrets['auth_key']

if 'status' not in st.session_state:
    st.session_state.status = 'submitted'
    st.session_state.save_location = ''
    st.session_state.transcription_completed = False
    
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
transcript_endpoint = 'https://api.assemblyai.com/v2/transcript'

headers = {
    "authorization": auth_key,
    "content-type": "application/json"    
}

st.set_page_config(page_title='LLM', 
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

@st.cache_data(ttl=600)
def download_audio(link):
    _id = link.strip()
    
    def get_vid(_id):
        # create object YouTube
        yt = YouTube(_id)
        audio_stream = yt.streams.filter(only_audio=True).first()
        print(audio_stream)
        return audio_stream
        
    # download the audio of the Youtube video locally
    audio_stream = get_vid(_id)
    download_path = './'
    file_name = audio_stream.download(output_path=download_path)   
    st.session_state.save_location = file_name 
    print('Saved audio to', st.session_state.save_location)
    return file_name
    
def read_file(filename):
    with open(filename, 'rb') as _file:
        while True:
            data = _file.read(CHUNK_SIZE)
            if not data:
                break
            yield data
            
def start_transcription():
    upload_response = requests.post(
        upload_endpoint, 
        headers=headers, 
        data=read_file(st.session_state.save_location)
    )
    
    audio_url = upload_response.json()['upload_url']
    st.write('Uploaded audio to', audio_url)
    
    transcript_request = {
        'audio_url': audio_url,
        'iab_categories': False,
        }

    transcript_response = requests.post(
        transcript_endpoint, 
        headers=headers, 
        json=transcript_request
    )    
    
    transcript_id = transcript_response.json()['id']
    polling_endpoint = transcript_endpoint + '/' + transcript_id
    
    return polling_endpoint

def main():
    with st.sidebar:
        # st.image('logo-250-transparente.png')
        st.header('Information')
        st.write("""
                This project was created with the goal of participating in the 'Streamlit LLM Hackathon 2023'. 
                
                
                .
                """)
        
        st.header('About')
        st.write('Details about this project can be found in https://github.com/htsnet/')
        
    # título
    title = f'LLM'
    st.title(title)
    
    subTitle = f'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    st.subheader(subTitle)

    # link
    link = st.text_input('Paste your Youtube video link and press Enter')
    
    if link != '':
        transcription = ''
        col1, col2 = st.columns(2)
       
        with col1:
            st.video(link)
            
        with col2:
            with st.spinner('Working... '):
                download_audio(link)
                pooling_endpoint = start_transcription() 
                
                while st.session_state.status != 'completed':
                    pooling_response = requests.get(
                        pooling_endpoint, 
                        headers=headers
                    )
                    st.session_state.status = pooling_response.json()['status']
                    transcription = pooling_response.json()['text']

            st.success('Done!')
            st.session_state.transcription_completed = True
            st.markdown(transcription)
        
        if st.session_state.transcription_completed:
            st.markdown(transcription)

if __name__ == '__main__':
	main()   