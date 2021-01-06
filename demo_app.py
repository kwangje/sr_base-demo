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


def main():
    st.title('화자식별 Demo App')
    st.text("Build with Streamlit and OpenCV")

    activities = ["데이터", "Demo"]
    choice = st.sidebar.selectbox("Select Activty", activities)


    if choice == '데이터':
        st.subheader("IPTV 음성검색 데이터")
        st.text('Noise 없는 실험실 녹음버전이며, 음성길이는 평균 5초 이내인 짧은 발화문입니다.')

        df = st.cache(pd.read_excel)(
            './audio_data/IPTV_TC.xlsx', nrows=140, engine='openpyxl')
        st.dataframe(df[df.columns[:4]])

        st.subheader("데이터 가공 예시")
        file_to_be_uploaded = st.file_uploader(
            "테스트할 Audio 파일 불러오기", type="wav")
        
        if file_to_be_uploaded is not None:
            audio_bytes = file_to_be_uploaded.read()
            st.audio(audio_bytes, format='audio/mp3')
        
        st.write(os.path.abspath(file_to_be_uploaded)
        #test_wav = preprocess_wav(test_path)
        
        #encoder = VoiceEncoder()
        #wavs = np.array(list(map(preprocess_wav, tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths), position=0)
        #st.write(speaker_wavs.get('김나영')[:3])
        
    elif choice == 'Demo':
        #st.subheader("Necessary for two libraries")
        st.text('Compute the similarity between spk embed and utt embed.')
        st.markdown(
            "Modeling: [pyannote-audio](https://github.com/pyannote/pyannote-audio/)")
        st.markdown(
            "Others: [Resemblyzer](https://github.com/resemble-ai/Resemblyzer/)")
        st.success(":bowtie:")


if __name__ == '__main__':
		main()	