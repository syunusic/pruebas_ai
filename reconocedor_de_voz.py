import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import speech_recognition as sr
import io
from pydub import AudioSegment

#tokenizer = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
#model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
tokenizer = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-spanish")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-spanish")


r = sr.Recognizer()

with sr.Microphone(sample_rate=16000) as source:
    print("Say something!")
    while True:
        audio = r.listen(source) # listen for the first phrase and extract it into audio data
        data = io.BytesIO(audio.get_wav_data()) # convert audio to wav
        clip = AudioSegment.from_file(data) # convert wav to mp3
        x = torch.FloatTensor(clip.get_array_of_samples()) # convert mp3 to tensor

        inputs = tokenizer(x, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
        logits = model(inputs).logits
        tokens = torch.argmax(logits, dim=-1)   # get the predicted token ids
        text = tokenizer.batch_decode(tokens) # convert ids to text

        print('You said: ', str(text).lower())