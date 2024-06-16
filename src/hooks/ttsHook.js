import { useEffect, useState, useRef } from "react";

import tts from '../Components/Tts';
const messages = [
  "Sidewalk might be ended please be careful",
]

const useTts = () => {
  const [bounce, setBounce] = useState(false);
  const [messageQueue, setMessageQueue] = useState(new Set());
  const lastTtsCall = useRef(Date.now());

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

  const debounceTts = (message) => {
    const now = Date.now();
    if (now - lastTtsCall.current > 3000) {
      tts(message);
      lastTtsCall.current = now;
    }
  };

  const handleTts = (messages) => {
    setBounce(true);
    const _messages = messages.map((v) => v);

    let interval = null;

    const speech = () => {
      tts(_messages.shift());
      if (_messages.length === 0) {
        clearInterval(interval);
        setBounce(false);
        return;
      }
    };

    interval = setInterval(speech, 4000);
  };

  const addMessage = (message) => {
    // setMessageQueue((prev) => new Set(prev).add(message));
    debounceTts(message);
  };

  return { addMessage, tts };


};
export { useTts };
