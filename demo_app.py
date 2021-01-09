# dependencies
import streamlit as st
import torch
import os

from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *
from itertools import groupby
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
cmaps = plt.colormaps()
mpl.rcParams['axes.unicode_minus'] = False 
plt.rcParams["font.family"] = 'NanumGothic'

from umap import UMAP
import plotly.express as px

########################################################################

def run_status():
	latest_iteration = st.empty()
	bar = st.progress(0)
	for i in range(100):
		latest_iteration.text(f'Percent Complete {i+1}')
		bar.progress(i + 1)
		time.sleep(0.1)
		st.empty()
  
def run_data():
    run_status()
    wav_fpaths = list(Path("audio_data", "iptv_uttr", "sample").glob("**/*.wav"))
    speakers = list(map(lambda wav_fpath: wav_fpath.parent.stem, wav_fpaths))
    wavs = np.array(list(map(preprocess_wav, tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths), position=0))))

def main():
    st.title('화자식별 Demo App')
    st.markdown("IPTV 화자식별 서비스에 필요한 주요기능을 ~~성능보장~~ 구현한 Prototype 입니다.")

    activities = ["USC", "SR"]
    choice = st.sidebar.selectbox("Select Activty", activities)

    if choice == 'USC':
        st.text('데모 상황: 1개월간 가입자 홍길동 고객님의 STB에 음성검색데이터 1580개 쌓여있음.')
        st.subheader("Unsupervised Speaker Clustering")
        
        st.markdown('1. 사전학습모델(UBM)을 통해 모든 음성데이터에 대한 저차원(256) 임베딩 벡터값을 구함')      
        btn = st.sidebar.button("Preprocessing")  
        if btn:
            run_data()
        else:
            pass
        
        st.markdown('2. threshold 를 설정하고, 00% 이상의 유사성 가진 음성끼리 군집화함.')

        #df = st.cache(pd.read_excel)('./audio_data/IPTV_TC.xlsx', nrows=140, engine='openpyxl')
        #st.dataframe(df[df.columns[:4]])
        #st.subheader("데이터 가공 예시")
        #pt = torch.hub.load('/home/kwangje/Desktop/sr-iptv-proto/resemblyzer/pretrained.pt')
        #print(pt)       
        
        #file_to_be_uploaded = file_selector()
        #file_var = AudioSegment.from_wav(file_to_be_uploaded) 
        #file_var.export('filename.wav', format='wav')       
        #st.audio(filename, format='audio/mp3')   
        
        file_to_be_uploaded = st.file_uploader(
            "테스트할 Audio 파일 불러오기", type="wav")
        
        if file_to_be_uploaded is not None:
            audio_bytes = file_to_be_uploaded.read()
            st.audio(audio_bytes, format='audio/mp3')
            
        sample = preprocess_wav('/home/kwangje/Desktop/sr-iptv-proto/audio_data/iptv_uttr/test/record_163357_pcm.wav')
        st.write(print(sample.tolist()))
        #test_embed = encoder_new.embed_utterance(sample)
        #st.write(test_embed)
        
        #st.write(os.path.abspath(file_to_be_uploaded))
        #test_wav = preprocess_wav(test_path)
        
        #encoder = VoiceEncoder()
        #wavs = np.array(list(map(preprocess_wav, tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths), position=0)
        #st.write(speaker_wavs.get('김나영')[:3])
        
    elif choice == 'SR':
        #st.subheader("Necessary for two libraries")
        st.text('Compute the similarity between spk embed and utt embed.')
        st.markdown(
            "Modeling: [pyannote-audio](https://github.com/pyannote/pyannote-audio/)")
        st.markdown(
            "Others: [Resemblyzer](https://github.com/resemble-ai/Resemblyzer/)")
        st.success(":bowtie:")


if __name__ == '__main__':
		main()	