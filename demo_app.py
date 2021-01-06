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

@st.cache(allow_output_mutation=True)
def file_selector(folder_path='./audio_data/iptv_uttr/test'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a test audio file', filenames)
    return os.path.join(folder_path, selected_filename)


def plot_projection_adv(embeds, speakers, ax=None, colors=None, markers=None, legend=True, 
                        title="", **kwargs):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))
        
    reducer = UMAP(n_neighbors=30, min_dist=0.2, **kwargs) # 2D projections
    projs = reducer.fit_transform(embeds)  # projs.shape
    speakers = np.array(speakers)
    
    for i, speaker in enumerate(np.unique(speakers)):
        speaker_projs = projs[speakers == speaker]
        marker = "o" if markers is None else markers[i]
        label = speaker if legend else None
        ax.scatter(*speaker_projs.T, cmap=plt.cm.Blues, 
                    marker=marker, label=label, alpha=0.7)

    if legend:
        ax.legend(title="Speakers", ncol=2, loc="upper right")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("auto")
    
    return projs

