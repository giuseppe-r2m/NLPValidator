NLPValidator helps in the execution of automatic tests for a text-to-text chain.

The application is still at initial stage, but it allows to define a hard-coded sentence
in amain(), execute TTS with Edge TTS (high-quality neural voices), mix the resulting
file(s) with noise at various SNR, then perform STT (FasterWhisper) and calculate CER/WER.

In future versions:
- text can be passed as a parameter
- noise filename no longer hardcoded, but passed as parameter
- more parameters for locale and gender of voices
- results written into a CSV rather than just printed
- possibility to invoke a filtering function before engaging the STT
- more diagnostics
