import { useEffect, useState } from "react";

import tts from '../Components/Tts';

const useTts = () => {
  const [bounce, setBounce] = useState(false);
  const [messageQueue, setMessageQueue] = useState(new Set());

  useEffect(() => {
    if (messageQueue.size === 0) return;
    if (bounce) return;
    const messages = Array.from(messageQueue.entries()).map(([key, value]) => {
      messageQueue.delete(key);
      return value;
    });
    handleTts(messages);
  }, [messageQueue, bounce]);

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
    setMessageQueue((prev) => new Set(prev).add(message));
  };

  return { addMessage };


};
export { useTts };
