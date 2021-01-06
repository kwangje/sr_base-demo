# dependencies
from typing import Optional, Union
import pandas as pd
from pydub import AudioSegment
import os
import streamlit as st
from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *
from itertools import groupby
from pathlib import Path
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"] = 'NanumGothic'

if __name__ == '__main__':
    st.title('Speaker Identification Demo')

    activities = ["Data", "About"]
    choice = st.sidebar.selectbox("Select Activty", activities)

    if choice == 'Data':
        st.subheader("dataset")
        st.text('test case')

        df = st.cache(pd.read_excel)(
            './audio_data/IPTV 음성검색_TC.xlsx', nrows=140)
        st.dataframe(df[df.columns[:4]])

        file_to_be_uploaded = st.file_uploader(
            "Choose an audio...", type="wav")
        if file_to_be_uploaded is not None:
            audio_bytes = file_to_be_uploaded.read()
            st.audio(audio_bytes, format='audio/mp3')

        st.write(speaker_wavs.get('김나영')[:3])

    elif choice == 'Demo':
        #st.subheader("Necessary for two libraries")
        st.text('Compute the similarity between spk embed and utt embed.')
        st.markdown(
            "Modeling: [pyannote-audio](https://github.com/pyannote/pyannote-audio/)")
        st.markdown(
            "Others: [Resemblyzer](https://github.com/resemble-ai/Resemblyzer/)")
        st.success(":bowtie:")

    # generator = instantiate_generator()
    # filename = file_selector()
    # ipd.Audio(filename)
    # st.write('selected `%s`' % filename)

    # wav_file = st.file_uploader("Upload Audio", type=['wav', 'flac', 'mp3'])

    # if wav_file is not None:
    #    audio_file = AudioSegment.from_wav(wav_file)
    #    st.write(type(audio_file))
