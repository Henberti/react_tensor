import { useEffect, useState, useRef } from "react";
import Tts from '../Components/Tts';

const useTts = () => {
  const [bounce, setBounce] = useState(false);
  const [messageQueue, setMessageQueue] = useState(new Set());
  const lastTtsCall = useRef(Date.now());
  const currentMessage = useRef("");

  useEffect(() => {
    if (messageQueue.size === 0) return;
    if (bounce) return;
    console.log("messageQueue", messageQueue);
    const messages = Array.from(messageQueue.entries()).map(([key, value]) => {
      messageQueue.delete(key);
      return value;
    });
    handleTts(messages);
  }, [messageQueue, bounce]);

  let voices = window.speechSynthesis.getVoices();
  const voice = voices[6];

  const debounceTts = (message, interrupt = false) => {
    const now = Date.now();
    if ((now - lastTtsCall.current > 3000 || interrupt) && message !== currentMessage.current) {
      if (interrupt) {
        currentMessage.current = message;
      }
      Tts(message, voice, interrupt);
      lastTtsCall.current = now;
    }
  };

  const handleTts = (messages) => {
    setBounce(true);
    const _messages = messages.map((v) => v);

    let interval = null;

    const speech = () => {
      const nextMessage = _messages.shift();
      currentMessage.current = nextMessage;
      Tts(nextMessage, voice);
      if (_messages.length === 0) {
        clearInterval(interval);
        setBounce(false);
        return;
      }
    };

    interval = setInterval(speech, 4000);
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
