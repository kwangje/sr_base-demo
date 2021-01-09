from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *
from itertools import groupby
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False 
plt.rcParams["font.family"] = 'NanumGothic'

'''
utterance embedding: umap projection
    - compare speech utterances between iptv speakers
    - metric: how similar their sounds
'''

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



wav_fpaths_3 = list(Path("audio_data", "iptv_uttr", "total").glob("**/*.wav"))
speakers_3 = list(map(lambda wav_fpath: wav_fpath.parent.stem, wav_fpaths_3))
wavs_3 = np.array(list(map(preprocess_wav, tqdm(wav_fpaths_3, "Preprocessing wavs", len(wav_fpaths_3), position=0))))
speaker_wavs_3 = {speaker: wavs_3[list(indices)] for speaker, indices in 
                  groupby(range(len(wavs_3)), lambda i: speakers_3[i])} 

## Compute the embeddings
encoder_3 = VoiceEncoder()
utterance_embeds_3 = np.array(list(map(encoder_3.embed_utterance, wavs_3)))

plot_projection_adv(utterance_embeds_3, speakers_3, title="Embedding projections")
#plt.show()
plt.savefig('demo01_#40.png')