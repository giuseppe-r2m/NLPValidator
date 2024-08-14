NLPValidator helps in the execution of automatic tests for a text-to-text chain.

How to use:

You need a recent python compiler like 3.12.
Then please install these libraries:
numpy
soundfile
faster_whisper
edge_tts
jiwer
re

pip install numpy soundfile faster_whisper edge_tts jiwer re

Note:
The application is still at initial stage, but it allows to define a hard-coded sentence in amain(), execute TTS with Edge TTS (high-quality neural voices), mix the resulting file(s) with noise at various SNR, then perform STT (FasterWhisper) and calculate CER/WER.

In future versions:
- texts can be passed as a parameter, or even better as a filename with a list of sentences
- noise filename no longer hardcoded, but passed as parameter
- more parameters for locale and gender of voices
- results written into a CSV rather than just printed
- possibility to invoke a filtering function before engaging the STT
- more diagnostics
