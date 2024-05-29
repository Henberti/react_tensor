const Tts = (message, voice) => {
    if (window.speechSynthesis) {
        const utterance = new SpeechSynthesisUtterance(message);
        utterance.voice = voice;
        window.speechSynthesis.speak(utterance);
    } else {
        console.log("Speech synthesis not supported.");
    }
};

export default Tts;