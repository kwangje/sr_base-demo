# dependencies
import plotly.express as px
import plotly.graph_objects as go
from umap import UMAP
import streamlit as st
from stqdm import stqdm
import torch
import time

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


########################################################################

@st.cache(suppress_st_warning=True)
def load_data():
    wav_fpaths = list(Path("audio_data", "iptv_uttr",
                           "sample1_wt_label").glob("**/*.wav"))
    speakers = list(map(lambda wav_fpath: wav_fpath.parent.stem, wav_fpaths))
    wavs = np.array(list(map(preprocess_wav, stqdm(
        wav_fpaths, "Preprocessing wavs", len(wav_fpaths), position=0))))
    st.success('Finished preprocessing')

    encoder = VoiceEncoder()
    utterance_embeds = np.array(
        list(map(encoder.embed_utterance, stqdm(wavs, "compute emb vec.", len(wavs)))))
    st.success('Finished embedding compute')

    speaker_wavs = {speaker: wavs[list(indices)] for speaker, indices in
                    stqdm(groupby(range(len(wavs)), lambda i: speakers[i]))}
    spk_embeds_total = np.array([encoder.embed_speaker(wavs[:len(wavs)])
                                 for wavs in stqdm(speaker_wavs.values())])
    st.success('Finished speaker embeddings')

    return speakers, wavs, encoder, utterance_embeds, speaker_wavs, spk_embeds_total


speakers, wavs, encoder, utterance_embeds, speaker_wavs, spk_embeds_total = load_data()


@st.cache(suppress_st_warning=True)
def umap_vis(utterance_embed):

    reducer = UMAP(n_neighbors=30, min_dist=0.2, n_components=3)
    projs = reducer.fit_transform(utterance_embed)  # projs.shape
    df = pd.DataFrame(projs)
    df["speaker"] = speakers
    df["성별"] = '남/여'
    df["연령대"] = '어린이 O/X'

    features = df.loc[:, [0, 1, 2]]
    umap_3d = UMAP(n_neighbors=30, min_dist=0.2, n_components=3)  # **kwargs
    proj_3d = umap_3d.fit_transform(features)

    fig_3d_raw = px.scatter_3d(
        proj_3d, x=0, y=1, z=2, opacity=0.7,
        # text=['point #{}'.format(i) for i in range(df.shape[0])]
    )
    fig_3d_raw.update_traces(marker_size=3)
    fig_3d_raw.update_layout(margin=dict(l=0, r=0, b=0, t=0))  # fig_3d.show()
    st.plotly_chart(fig_3d_raw, user_container_width=True)

    fig_3d = px.scatter_3d(
        proj_3d, x=0, y=1, z=2,
        color=df.speaker, labels={'color': 'speaker'}, opacity=0.7,
        # text=['point #{}'.format(i) for i in range(df.shape[0])]
    )
    fig_3d.update_traces(marker_size=3)
    fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0))  # fig_3d.show()
    st.plotly_chart(fig_3d, user_container_width=True)


def write_preprocess_page():
    st.write("""
1. Audio 전처리
    - Audio to Waveform, 16K resampling, Normalize Volume, Shorten long silence
2. 사전학습모델(UBM)을 통해 모든 음성데이터에 대한 저차원(256) 임베딩 벡터값을 구함
    - 참고로 본 기술 전반의 core 는 UBM 제작에 있습니다.
3. Manifold Learning 수행
    - 임베딩 벡터값이 Clustering을 하기에 적합한지 판단하기 위해 UMAP 시각화 실시
4. 관리자페이지 역할
    - 임베딩벡터 파라미터 조정, 부적합한 음성데이터 제거, UBM 재학습 등 판단
5. 최종 군집 개수 결정 및 화자식별 파이프라인으로 결과 전달
    - 군집 개수(i.e 발화자 인원 수)를 결정하고, (optional) 군집에 포함되지 않는 데이터는 제거함
"""
             )

########################################################################


st.title('화자식별 Demo App')
st.markdown("IPTV 화자식별 서비스에 필요한 주요기능을 ~~성능보장~~ 구현한 Prototype 입니다.")

activities = ["EDA", "USC", "SR"]
st.sidebar.title("Menu")
choice = st.sidebar.selectbox("Select Activty", activities)

if choice == 'EDA':
    st.subheader('EDA')
    st.text("""
    본 Demo App 성능테스트는 20년 7월에 임직원 대상으로 녹음한 IPTV 음성검색 발화문 입니다.
    (41명의 발화자, 개별 140개 발화문입니다.)
    """
            )
    st.write("발화문 script")

    df = st.cache(pd.read_excel)(
        './audio_data/IPTV_TC.xlsx', nrows=140, engine='openpyxl')
    st.dataframe(df[df.columns[:4]])
    file_to_be_uploaded = st.file_uploader(
        "테스트할 Audio 파일 불러오기", type="wav")
    if file_to_be_uploaded is not None:
        audio_bytes = file_to_be_uploaded.read()
        st.audio(audio_bytes, format='audio/mp3')

    st.text("""
    본 Demo App 에서 사용한 임베딩 벡터
    현재 10명 발화자, 총 1590개 발화문이며, 다른 데이터로 테스트할 경우, path 바꾼 후 재실행!
    """
            )
    df_new = pd.DataFrame(list(zip(speakers, wavs)),
                          columns=['speakers', 'wavs'])
    st.dataframe(df_new)

elif choice == 'USC':
    st.subheader('Unsupervised Speaker Clustering')
    write_preprocess_page()
    btn = st.button("UMAP")
    if btn:
        umap_vis(utterance_embeds)
    else:
        pass


elif choice == 'SR':
    st.subheader('Speaker Reocognition')
    st.text('※ 본 Demo APP 에서는 USC를 통해 "군집 개수=10"으로 결정함.')

    st.write("""
    - 군집별 speaker embeddings 생성: 임베딩 벡터 평균 및 정규화
        - presumably from the same speaker
    """
             )

    st.success('Compute the embedding of a collection of wavs')

    st.write("""
    Unknown input에 대해, 10개 군집 중 어느 곳에 속할지 (or 속하지 않는지) 유사도 계산으로 판단함
    """)

    def hist(speaker_wavs, spk_embeds_total, test_embed):
        spk_sim_matrix = np.inner(spk_embeds_total, test_embed)

        labels = [i for i in speaker_wavs.keys()]
        stats = dict(zip(labels, spk_sim_matrix))

        fig = go.Figure([go.Bar(x=labels, y=spk_sim_matrix)])

        result = max(stats, key=stats.get)

        fig.update_layout(margin=dict(l=5, r=5, b=5, t=5))
        st.plotly_chart(fig)
        return stats, result

    file_to_be_uploaded = st.file_uploader(
        "테스트할 Audio 파일 불러오기", type="wav")

    test_wav = ["김나영1", "김나영2", "이윤규_Unk_1",
                "이윤규_Unk_2", "하미향1_unk", "하미향2_unk"]
    choice = st.selectbox("select a sample", test_wav)

    if choice == '김나영1':
        audio_file = open('./audio_data/iptv_uttr/test/김나영1.wav', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')

        test_path = Path("audio_data", "iptv_uttr", "test", "김나영1.wav")
        test_wav = preprocess_wav(test_path)
        test_embed = encoder.embed_utterance(test_wav)

        stats, result = hist(speaker_wavs, spk_embeds_total, test_embed)
        if stats[result] >= 0.8:
            st.write('식별한 화자는 {}'.format(str(result)))
        else:
            st.write('80% 미만의 유사도로 추정되는 화자 없음.')

    elif choice == '김나영2':
        audio_file = open('./audio_data/iptv_uttr/test/김나영2.wav', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')

        test_path = Path("audio_data", "iptv_uttr", "test", "김나영2.wav")
        test_wav = preprocess_wav(test_path)
        test_embed = encoder.embed_utterance(test_wav)

        stats, result = hist(speaker_wavs, spk_embeds_total, test_embed)
        if stats[result] >= 0.8:
            st.write('식별한 화자는 {}'.format(str(result)))
        else:
            st.write('80% 미만의 유사도로 추정되는 화자 없음.')

    elif choice == '이윤규_Unk_1':
        audio_file = open('./audio_data/iptv_uttr/test/이윤규1_unk.wav', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')

        test_path = Path("audio_data", "iptv_uttr", "test", "이윤규1_unk.wav")
        test_wav = preprocess_wav(test_path)
        test_embed = encoder.embed_utterance(test_wav)

        stats, result = hist(speaker_wavs, spk_embeds_total, test_embed)
        if stats[result] >= 0.8:
            st.write('식별한 화자는 {}'.format(str(result)))
        else:
            st.write('80% 미만의 유사도로 추정되는 화자 없음.')

    elif choice == '이윤규_Unk_2':
        audio_file = open('./audio_data/iptv_uttr/test/이윤규2_unk.wav', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')

        test_path = Path("audio_data", "iptv_uttr", "test", "이윤규2_unk.wav")
        test_wav = preprocess_wav(test_path)
        test_embed = encoder.embed_utterance(test_wav)

        stats, result = hist(speaker_wavs, spk_embeds_total, test_embed)
        if stats[result] >= 0.8:
            st.write('식별한 화자는 {}'.format(str(result)))
        else:
            st.write('80% 미만의 유사도로 추정되는 화자 없음.')

    elif choice == '하미향1_unk':
        audio_file = open('./audio_data/iptv_uttr/test/하미향1_unk.wav', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')

        test_path = Path("audio_data", "iptv_uttr", "test", "하미향1_unk.wav")
        test_wav = preprocess_wav(test_path)
        test_embed = encoder.embed_utterance(test_wav)

        stats, result = hist(speaker_wavs, spk_embeds_total, test_embed)
        if stats[result] >= 0.8:
            st.write('식별한 화자는 {}'.format(str(result)))
        else:
            st.write('80% 미만의 유사도로 추정되는 화자 없음.')

    elif choice == '하미향2_unk':
        audio_file = open('./audio_data/iptv_uttr/test/하미향2_unk.wav', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')

        test_path = Path("audio_data", "iptv_uttr", "test", "하미향2_unk.wav")
        test_wav = preprocess_wav(test_path)
        test_embed = encoder.embed_utterance(test_wav)

        stats, result = hist(speaker_wavs, spk_embeds_total, test_embed)
        if stats[result] >= 0.8:
            st.write('식별한 화자는 {}'.format(str(result)))
        else:
            st.write('80% 미만의 유사도로 추정되는 화자 없음.')
