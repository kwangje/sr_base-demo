import streamlit as st

from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *

from itertools import groupby
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from pydub import AudioSegment
from typing import Optional, Union



@st.cache(allow_output_mutation=True)
def file_selector(folder_path='./audio_data/iptv_uttr/test'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a test audio file', filenames)
    return os.path.join(folder_path, selected_filename)



if __name__ == '__main__':
    st.title('Speaker Identification Demo')
    st.text('Compute the similarity between spk embed and utt embed.')
    
    #generator = instantiate_generator()
    filename = file_selector()
    st.write('selected `%s`' % filename)
    
    
    #wav_file = st.file_uploader("Upload Audio", type=['wav', 'flac', 'mp3'])

    #if wav_file is not None:
    #    audio_file = AudioSegment.from_wav(wav_file)        
    #    st.write(type(audio_file))

