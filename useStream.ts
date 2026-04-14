"use client";

import { useRef } from "react";
import { useStore } from "../store/useStore";

export const useStream = () => {
  const { appendStream, addLog } = useStore();

  const eventRef = useRef<EventSource | null>(null);
  const bufferRef = useRef<string>("");
  const fullResponse = useRef<string>("");

  const flushInterval = useRef<ReturnType<typeof setInterval> | null>(null);

  /* =========================
     FLUSH BUFFER
  ========================= */
  const startFlush = () => {
    if (flushInterval.current) return;

    flushInterval.current = setInterval(() => {
      if (!bufferRef.current) return;

      appendStream(bufferRef.current);
      bufferRef.current = "";
    }, 40); // ⚡ slightly faster
  };

  const stopFlush = () => {
    if (flushInterval.current) {
      clearInterval(flushInterval.current);
      flushInterval.current = null;
    }
  };

  /* =========================
     CONNECT SSE
  ========================= */
  const connect = () => {
    if (eventRef.current) return;

    addLog("🔌 Connecting SSE...");

    const es = new EventSource("http://localhost:8000/stream");
    eventRef.current = es;

    startFlush();

    es.onopen = () => {
      addLog("✅ SSE Connected");
    };

    es.onmessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);

        // 🧠 THINKING
        if (data.type === "thinking") {
          useStore.getState().setThinking(data.content);
        }

        // ⚡ STREAM TOKENS
        if (data.token) {
          bufferRef.current += data.token;
          fullResponse.current += data.token;
        }

        // ✅ DONE
        if (data.done) {
          if (bufferRef.current) {
            appendStream(bufferRef.current);
            bufferRef.current = "";
          }

          const finalText = fullResponse.current.trim();

          if (finalText) {
            useStore.getState().replaceLastMessage(finalText);
          }

          fullResponse.current = "";

          es.close();
          eventRef.current = null;
          stopFlush();
        }

      } catch {
        addLog("❌ Parse error");
      }
    };

    es.onerror = () => {
      addLog("⚠ Stream error");

      es.close();
      eventRef.current = null;
      stopFlush();
    };
  };

  /* =========================
     SEND QUERY
  ========================= */
  const sendQuery = async (query: string) => {
    try {
      addLog("📤 Sending query...");

      // 🧠 RESET
      useStore.getState().setThinking("");
      useStore.getState().clearStream();

      // 🔥 SHOW INSTANT FEEDBACK (IMPORTANT)
      useStore.getState().addMessageToCurrent({
        id: crypto.randomUUID(),
        role: "ai",
        content: "🧠 Thinking...",
      });

      await fetch("http://localhost:8000/query", {
        method: "POST",
        body: new URLSearchParams({ query }),
      });

      connect();

    } catch {
      addLog("❌ Failed to send query");
    }
  };

  return { sendQuery };
};