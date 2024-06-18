import { useRef } from "react";
import Tts from "../Components/Tts";

const useTts = () => {
  const lastTtsCall = useRef(Date.now());
  const currentMessage = useRef("");

  let voices = window.speechSynthesis.getVoices();
  const voice = voices[6];

  const debounceTts = (message, interrupt = false) => {
    const now = Date.now();
    if (
      (now - lastTtsCall.current > 3000 || interrupt) &&
      message !== currentMessage.current
    ) {
      if (interrupt) {
        currentMessage.current = message;
      }
      Tts(message, voice, interrupt);
      lastTtsCall.current = now;
    }
  };

  const addMessage = (prediction, type, critical = false) => {
    let message;
    if (type === "obstacle") {
      const messages = [
        "There is a " + prediction + " in front of you.",
        "You are walking toward a " + prediction,
        "There is a " + prediction + " in your way. Please adjust your path.",
      ];
      const randomIndex = Math.floor(Math.random() * messages.length);
      message = messages[randomIndex];
    } else if (type === "road") {
      message = prediction;
    } else {
      message = prediction;
    }
    debounceTts(message, critical);
  };

  return { addMessage, Tts };
};

export { useTts };
