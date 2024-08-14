#!/usr/bin/env python3
from typing import List, Dict, Any
import asyncio
import random
import io
import numpy as np
import soundfile
from faster_whisper import WhisperModel
import edge_tts
from edge_tts import VoicesManager
import os
import jiwer
import re

model_size:str = "large-v3"
#model_size:str = "medium"
#model_size:str = "tiny"
print(f"\nLoading FasterWhisper model {model_size}, this may get a long time.")
model = WhisperModel(model_size, device="auto", compute_type="int8")
# Level of printed info. 0 no info, 1 basic info, 2 detailed info
infoLevel: int = 0

async def callTTS(text: str, voiceName: str) -> bytes:
    """
    Invokes Edge's TTS service and returns a byte stream
    containing the voice.
    text = text to be converted
    voiceName = ShortName attribute of one of the Edge TTS
    voices.
    Return: stream of bytes with the TTS result.
    """
    result: bytearray = bytearray()
    communicate = edge_tts.Communicate(text, voiceName)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            result.extend(chunk["data"])
    return bytes(result)

async def selectTTSVoices(sample: int, locale: str = "en", gender: str = "") -> List[Dict[str, Any]]:
    """
    Returns a list containing short names of Edge TTS voices
    randomly selected among those available.
    Input:
        <sample> = number of different voices to select
        <locale> = locale to use for voice selection, defaults to "en"
        <gender> = "Male" or "Female", case sensitive, to retrieve only one voice type.
                    Default to empty (both).
    Return:
        a List[] containing voice data. For the contents, refer to VoiceManager,
        but in general the attribute needed for voice generation is "ShortName".
        The list size is the minimum between <sample> and the number of
        TTS voices available.
    """

    if (infoLevel >= 2):
        print(f"Asking Edge TTS {sample} voices among those available for locale '{locale}'")
    voices = await VoicesManager.create()
    if (gender == ''):
        voiceList = voices.find(Language=locale)
    else:
        voiceList = voices.find(Language=locale,Gender=gender)
    voiceSelection = random.sample(voiceList,min(sample,len(voiceList)))
    return voiceSelection

def bytesToAudioStream(audio_bytes):
    with io.BytesIO(audio_bytes) as input:
        data, sample_rate = soundfile.read(input)
    return data, sample_rate

def audioStreamToBytes(data, sample_rate):
    with io.BytesIO() as output:
        soundfile.write(output, data, sample_rate, format='WAV')
        audio_bytes = output.getvalue()
    return audio_bytes

def resampleAudioStream(data, originalSR, newSR):
    """
    Resample an array containing an audio stream.
    The function doesn't implement safety checks on parameters.
    Input:
    <data> = audio data to convert
    <originalSR> = current sample rate in Hz
    <newSR> = target sample rate in Hz
    Return:
    a tuple with <resampled data>, <target sample rate>
    """
    if (originalSR != newSR):
        original_length = len(data)
        new_length = int(np.round(original_length * newSR / originalSR))
        # linspace() creates two sequences of equally distant numbers, used to interpolate
        original_time = np.linspace(0, original_length - 1, original_length, endpoint=False) / originalSR
        new_time = np.linspace(0, new_length - 1, new_length, endpoint=False) / newSR
        # Perform linear interpolation to resample the audio data
        resampled_data = np.interp(new_time, original_time, data)
        return resampled_data, newSR
    else:
        # Nothing to convert
        return data, originalSR

def normalizeAudioStream(data):
    max_val = np.max(np.abs(data))
    if max_val == 0:
        return data  # Avoid division by zero
    return data / max_val

def mixAudioAndNoise(voiceData, noiseData, SNR_dB: float, voiceSR: int, noiseSR: int):
    if voiceSR != noiseSR:
        noiseData, noiseSR = resampleAudioStream(noiseData, noiseSR, voiceSR)
    # Truncate the longest array to the shorter
    if (len(noiseData) > len(voiceData)):
        noiseData = noiseData[:len(voiceData)]
    else:
        voiceData = voiceData[:len(noiseData)]
    # normalize
    voiceData = normalizeAudioStream(voiceData)
    noiseData = normalizeAudioStream(noiseData)

    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (SNR_dB / 10)
    # Calculate the noise scaling factor
    noise_scale = np.sqrt(1 / snr_linear)
    # Mix the audio signals
    mixedData = voiceData + (noiseData * noise_scale)
    mixedData = normalizeAudioStream(mixedData)
    return mixedData, voiceSR

async def createTTSFiles(text: str, iterations: int = 1, locale: str = "en", gender:str = "") -> List[str]:
    """
    Async function, call it with await createTTSFIles(...)
    Converts the input <text> into voice by calling Edge TTS, then saves the
    result as an audio WAV file sampled at 24 kHz.
    If <iteration> is more than 1, then the process is repeated, selecting each
    time a different voice.
    File names follow this format:
        <Edge short voice name>.wav
    (for example: "en-US-ChristopherNeural.wav").
    
    Input:
        <text> = text to convert with TTS
        <iterations> = how many files with diffent voices should be created, default 1.
        <locale> = locale to use for language selection, defaults to "en"
        <gender> = "male" or "female" to force only one type of voice. Leave empty for both.
    Return:
        a List[str] containing file names
    """
    voiceStream: bytes
    returnList: List[str] = []

    #   - creates a selection of TTS voices
    voiceSelection = await selectTTSVoices(iterations, locale, gender)
    for voice in voiceSelection:
        voiceName = voice['ShortName']
    #   - creates the TTS of the text for each voice, and saves it to a file
        # voiceStream is a <bytes> object with audio encoded in Opus
        if (infoLevel >= 1):
            print(f"Preparing TTS file named {voiceName}")
        voiceStream = await callTTS(text,voiceName)
        # Let's extract audio and sample rate
        voiceData, voiceSR = bytesToAudioStream(voiceStream)
        # resample everything at 24000 Hz
        voiceData, voiceSR = resampleAudioStream(voiceData,voiceSR,24000)
        voiceFileName = f"{voiceName}.wav"
        with open(voiceFileName,"wb") as file:
            # convert back to byte stream and write
            file.write(audioStreamToBytes(voiceData,voiceSR))
            returnList.append(voiceFileName)
    return returnList

def createNoisyFiles(fileList: List[str], noiseFile: str, SNRList: List[float]) -> List[str]:
    """
    Creates "noisy" versions of existing files, by mixing the original audio
    with noise at various SNR.
    The function reads each file in <fileList>, then mixes the audio stream
    with that contained in <noiseFile>, using each element in <SNRList> as SNR.
    For example, if the list contains [-10,-5-,0,5,10] then the function will
    create five new files for each entry in <fileList>, mixing the noise at
    -10 dB, -5 dB and so on. Note that a negative SNR means that the noise is
    larger than the signal.
    All files are saved as wav at 24kHz with this format:

        <name in fileList>_<sign symbol><SNR>.wav

    where <sign_symbol> is P for non-negative SNR, M otherwise.
    (for example: "en-GB-SoniaNeural_P005.wav")

    Input:
    <fileList> = list of names of files to read. Names can contain paths.
    <noiseFile> = name of the file containing noise.
    <SNRList> = list of SNR to use during the generation of files
    Return:
        a List[str] containing file names
    """
    returnList: List[str] = []

    # First, acquire the noise from <noiseFile>. This is not very efficient,
    # because it is repeated at each function call. Maybe a better possibility
    # could be to directly pass the audio samples array and the sampling rate.
    noiseStream: bytes = open(noiseFile,"rb").read()
    noiseData, noiseSR = bytesToAudioStream(noiseStream)
    # resample everything at 24000 Hz
    noiseData, noiseSR = resampleAudioStream(noiseData,noiseSR,24000)

    # Now create noisy files with different SNR
    for voiceFile in fileList:
        # extracts directory, filename and extension. This will be useful later
        dir, name = os.path.split(voiceFile)
        name, ext = os.path.splitext(name)
        for SNR in SNRList:
            if (infoLevel >= 2):
                print(f"Mixing {voiceFile} with noise at {SNR} dB")
            voiceStream: bytes = open(voiceFile,"rb").read()
            voiceData, voiceSR = bytesToAudioStream(voiceStream)
            # resample everything at 24000 Hz
            voiceData, noiseSR = resampleAudioStream(voiceData,voiceSR,24000)
            # The mixer operates directly with data and sampling rates
            mixed_data, mixed_sample_rate = mixAudioAndNoise(voiceData, noiseData, SNR, voiceSR, noiseSR)
            noisyFileName = f"{dir}{name}_{'P' if SNR >= 0 else 'M'}{abs(SNR):03}{ext}"
            if (infoLevel >= 2):
                print(f"Writing file in {noisyFileName}")
            with open(noisyFileName,"wb") as file:
                # But to write them in the file, we need to convert back to bytes
                file.write(audioStreamToBytes(mixed_data, mixed_sample_rate))
                returnList.append(noisyFileName)
    return returnList

def DUT(data: bytes, locale: str = "en"):
    # Invoke FasterWhisper to convert the audio, then returns the sentence
    if (infoLevel >= 1):
        print("Engaging the STT converter",end='')
    segments, info = model.transcribe(data, beam_size=5, language=locale)
    result: str = ""
    for segment in segments:
        result = result + segment.text
        if (infoLevel >= 2):
            print(".",end='')
    if (infoLevel >= 1):
        print()
    return result

async def amain() -> None:

#    TEXT = "I think it's a negative bump"
    TEXT = "Look, that seems a scratch on the surface"
    noiseFilename = "Audio cabina revisione_2_(1_min).wav"
    # Different SNR for generating noisy audio. 10 dB means that the
    # voice is 10 dB louder, -10 dB that the noise is higher.
    SNRList: List[float] = [5,0,-5,-10]
    locale: str = "en"
    iterations: int = 1
    gender: str = "Male"
    ver: str = "0.1"

    print(f"Rare2 NLP Validator v{ver}\nProcess starting...")
    # creates and writes audio files with different voices and the same text
    fileList = await createTTSFiles(TEXT,iterations,locale,gender)
    # mixes each file with noise
    noisyFileList = createNoisyFiles(fileList,noiseFilename,SNRList)
    # for each noise file, invoke the STT chain and writes down results
    for noisyFile in noisyFileList:
        noiseStream: bytes = open(noisyFile,"rb").read()
        noiseData, noiseSR = bytesToAudioStream(noiseStream)
        result = DUT(noiseData,locale)
        # prepare measurement texts in small caps and with no punctuation
        _TEXT = re.sub(r'[.,;:!?]', '', TEXT.lower())
        _result = re.sub(r'[.,;:!?]', '', result.lower())
        print(f"{noisyFile} \
              \n\tCER:{jiwer.cer(_TEXT,_result):.3}, \
              WER:{jiwer.wer(_TEXT,_result):.3} \
              \n\tcleaned ref:\t{_TEXT} \
              \n\tcleaned hyp:\t{_result}"
        )
    print("Process completed.")

if __name__ == "__main__":
    asyncio.run(amain())
