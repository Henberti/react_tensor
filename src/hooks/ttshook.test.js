import { renderHook, act } from '@testing-library/react';
import { useTts } from '../hooks/ttsHook';

test('test message queue', () => {
    const { result } = renderHook(() => useTts());

    act(() => {
        result.current.addMessage("Hello");
    });

    // You can add more assertions here to verify the behavior
    // For example, you can check if the message was added to the queue
});
