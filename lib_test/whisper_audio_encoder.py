import whisper
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import soundfile as sf

from tqdm.notebook import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CustomAudioDataset(Dataset):
      def __init__(self, csv_path):
            self.data = pd.read_csv(csv_path, header=[0, 1])
            self.text_data = self.data['text_data'][' '].values
            self.wav_dir = self.data['wav_dir'][' '].values

      def __len__(self):
            return len(self.data)

      def __getitem__(self, idx):
            try:
                audio_input, sample_rate = sf.read(self.wav_dir[idx])
                audio = whisper.pad_or_trim(audio_input)
                audio = torch.tensor(audio, dtype = torch.float32)
                mel = whisper.log_mel_spectrogram(audio)
            except:
                return None
            return (mel, self.text_data[idx])

def my_collate(batch):
    batch_size = len(batch)
    batch = list(filter(lambda x: x is not None, batch))

    if batch_size > len(batch):
        db_len = len(dataset)
        diff = batch_size - len(batch)
        while diff != 0:
            a = dataset[np.random.randint(0, db_len)]
            if a is None:
                continue
            batch.append(a)
            diff -= 1

    return torch.utils.data.dataloader.default_collate(batch)

dataset = CustomAudioDataset('balance_train.csv')

loader = torch.utils.data.DataLoader(dataset, collate_fn=my_collate, batch_size=16)

model = whisper.load_model("base")

options = whisper.DecodingOptions(language="ko", without_timestamps=True)

hypotheses = []
references = []

for mels, texts in tqdm(loader):
    mels = mels.to('cuda')
    results = model.decode(mels, options)
    hypotheses.extend([result.text for result in results])
    references.extend(texts)

data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))
data