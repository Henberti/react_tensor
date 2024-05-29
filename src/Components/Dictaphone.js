import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import React, { useState, useEffect } from 'react';
import tts from './Tts';

const Dictaphone = ({ setStarted }) => {
  const [commandProcessed, setCommandProcessed] = useState(false); 

  const {
    transcript,
    listening,
    resetTranscript,
    browserSupportsSpeechRecognition
  } = useSpeechRecognition();

  useEffect(() => {
    const handleStartCommand = () => {
      setStarted(true);
      let voices = window.speechSynthesis.getVoices();
      const voice = voices.length > 24 ? voices[24] : voices[0]; 
      tts("Started", voice);
      setCommandProcessed(true); 
      resetTranscript(); 
    };

    if ((transcript.includes("hello") || transcript.includes("Hello")) && !commandProcessed) {
      handleStartCommand();
    }
  }, [transcript, commandProcessed, setStarted]); 

  if (!browserSupportsSpeechRecognition) {
    return <span>Browser doesn't support speech recognition.</span>;
  }

  return (
    <div>
      <p>Microphone: {listening ? 'on' : 'off'}</p>
      <button onClick={() => SpeechRecognition.startListening({ language: 'en-US' })}>Start</button>
      <button onClick={SpeechRecognition.stopListening}>Stop</button>
      <button onClick={resetTranscript}>Reset</button>
      <p>{transcript}</p>
    </div>
  );
};

export default Dictaphone;
