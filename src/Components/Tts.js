const Tts = (message, voice, interrupt = false) => {
    if (window.speechSynthesis) {
      if (interrupt) {
        window.speechSynthesis.cancel(); 
      }
      const utterance = new SpeechSynthesisUtterance(message);
      // utterance.voice = voice;
      window.speechSynthesis.speak(utterance);
    } else {
      console.log("Speech synthesis not supported.");
    }
  };
  
  export default Tts;
  