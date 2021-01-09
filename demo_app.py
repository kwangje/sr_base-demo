# dependencies
import streamlit as st
from stqdm import stqdm
import torch, time

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

@st.cache(suppress_st_warning=True)
def umap_vis():
    wav_fpaths = list(Path("audio_data", "iptv_uttr", "sample1_wt_label").glob("**/*.wav"))
    speakers = list(map(lambda wav_fpath: wav_fpath.parent.stem, wav_fpaths))
    wavs = np.array(list(map(preprocess_wav, stqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths), position=0))))
    st.success('Finished preprocessing')
    
    encoder = VoiceEncoder()
    utterance_embeds = np.array(list(map(encoder.embed_utterance, stqdm(wavs, "compute emb vec.", len(wavs)))))
    st.success('Finished embedding compute')
    
    reducer = UMAP(n_neighbors=30, min_dist=0.2, n_components=3) 
    projs = reducer.fit_transform(utterance_embeds)  # projs.shape 
    df = pd.DataFrame(projs)
    df["speaker"] = speakers
    df["성별"] = '남/여'
    df["연령대"] = '어린이 O/X'
    features = df.loc[:, [0,1,2]]
    umap_3d = UMAP(n_neighbors=30, min_dist=0.2, n_components=3) # **kwargs
    proj_3d = umap_3d.fit_transform(features) 
    fig_3d = px.scatter_3d(
        proj_3d, x=0,y=1,z=2,
        color=df.speaker, labels={'color': 'speaker'}, opacity=0.7)
    fig_3d.update_traces(marker_size=2) 
    fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0)) # fig_3d.show() 
    st.plotly_chart(fig_3d, user_container_width=True)

########################################################################   

st.title('화자식별 Demo App')
st.markdown("IPTV 화자식별 서비스에 필요한 주요기능을 ~~성능보장~~ 구현한 Prototype 입니다.")
    
activities = ["EDA", "USC", "SR"]
choice = st.sidebar.selectbox("Select Activty", activities)

if choice == 'EDA':
    st.header('EDA')     
    st.text('Demo 테스트데이터: 41명 140개 IPTV 음성발화 녹음본')
    st.text('※ 특정 화자는 반복 녹음으로 140개 이상임')
    df = st.cache(pd.read_excel)('./audio_data/IPTV_TC.xlsx', nrows=140, engine='openpyxl')
    st.dataframe(df[df.columns[:4]])
    file_to_be_uploaded = st.file_uploader(
        "테스트할 Audio 파일 불러오기", type="wav")
    if file_to_be_uploaded is not None:
        audio_bytes = file_to_be_uploaded.read()
        st.audio(audio_bytes, format='audio/mp3') 
    
elif choice == 'USC':
    st.header("Unsupervised Speaker Clustering")     
    #st.subheader('(2) 전처리: Audio to Waveform, 16K resampling, Normalize Volume, Shorten long silence')     
    #st.text('※ 데모 상황: 1개월간 가입자 홍길동 고객님의 STB에 음성검색데이터 1590개 쌓여있음.')           
    #st.text('※ 속도 문제로 PosixPath로 데이터I/O 처리 중: sample2 테스트할 때 재실행하기')
    #st.subheader('(3) 사전학습모델(UBM)을 통해 모든 음성데이터에 대한 저차원(256) 임베딩 벡터값을 구함')
    st.subheader('Manifold Learning: 임베딩 벡터값이 Clustering을 하기에 적합한지 판단하기 위해 UMAP 시각화 실시')
    st.text('관리자페이지에서 각 STB 별로 결과 확인하기 위한 용도.')
    btn = st.button("UMAP")
    if btn:
        umap_vis()
    else:
        pass
    
    # st.markdown('(5) threshold 를 설정하고, 00% 이상의 유사성 가진 음성끼리 군집화함.')

    
elif choice == 'SR':
    st.text('Compute the similarity between spk embed and utt embed.')
    st.markdown(
        "Modeling: [pyannote-audio](https://github.com/pyannote/pyannote-audio/)")
    st.markdown(
        "Others: [Resemblyzer](https://github.com/resemble-ai/Resemblyzer/)")
    st.success(":bowtie:")
