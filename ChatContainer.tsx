
"use client";

import { useStore } from "../../store/useStore";
import { motion } from "framer-motion";
import { useEffect, useRef, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export default function ChatContainer() {
  const { conversations, currentChatId, stream, error } = useStore();

  const bottomRef = useRef<HTMLDivElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const [mounted, setMounted] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  const currentChat = useMemo(
    () => conversations.find((c) => c.id === currentChatId),
    [conversations, currentChatId]
  );

  const messages = currentChat?.messages ?? [];
  const isStreaming = Boolean(stream);

  useEffect(() => {
    if (!autoScroll) return;
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, stream]);

  const handleScroll = () => {
    const el = containerRef.current;
    if (!el) return;

    const isNearBottom =
      el.scrollHeight - el.scrollTop - el.clientHeight < 100;

    setAutoScroll(isNearBottom);
  };

  const copyText = async (text: string, id: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 1500);
    } catch {
      console.error("Copy failed");
    }
  };

  /* =========================
     ✅ FIXED MESSAGE RENDER
  ========================= */
  const renderedMessages = useMemo(() => {
    if (!mounted) return null;

    return messages.map((msg, i) => {
      const id = `${i}-${msg.role}`;

      const isLast = i === messages.length - 1;

      // 🔥 FIX: hide last AI message while streaming
      if (isStreaming && msg.role === "ai" && isLast) {
        return null;
      }

      return (
        <motion.div
          key={id}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          className={`flex ${
            msg.role === "user" ? "justify-end" : "justify-start"
          }`}
        >
          <div className="flex gap-2 max-w-[75%]">

            {msg.role === "ai" && (
              <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center text-xs">
                🤖
              </div>
            )}

            <div
              className={`rounded-2xl px-4 py-3 text-sm relative group ${
                msg.role === "user"
                  ? "bg-blue-600 text-white"
                  : "bg-[#1a1a1a] text-gray-200"
              }`}
            >
              <div className="prose prose-invert max-w-none text-sm leading-relaxed">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {msg.content}
                </ReactMarkdown>
              </div>

              {msg.role === "ai" && (
                <button
                  onClick={() => copyText(msg.content, id)}
                  className="absolute top-1 right-2 text-xs opacity-0 group-hover:opacity-100 bg-black/50 px-2 py-0.5 rounded"
                >
                  {copiedId === id ? "✔" : "Copy"}
                </button>
              )}
            </div>

            {msg.role === "user" && (
              <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-xs">
                👤
              </div>
            )}
          </div>
        </motion.div>
      );
    });
  }, [messages, copiedId, mounted, isStreaming]);

  if (!mounted) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500 text-sm">
        Loading...
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto space-y-6 px-4 py-6"
      >

        {/* ERROR */}
        {error && (
          <div className="bg-red-500/10 border border-red-500/30 text-red-300 text-sm px-4 py-3 rounded-xl">
            ⚠️ {error}
          </div>
        )}

        {/* EMPTY */}
        {messages.length === 0 && !isStreaming && (
          <div className="flex items-center justify-center h-full text-gray-500 text-sm">
            💬 Start a conversation
          </div>
        )}

        {/* MESSAGES */}
        {renderedMessages}

        {/* =========================
           🔥 STREAM (ONLY ONE)
        ========================= */}
        {isStreaming && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex justify-start"
          >
            <div className="flex gap-2 max-w-[75%]">

              <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center text-xs">
                🤖
              </div>

              <div className="rounded-2xl px-4 py-3 text-sm bg-[#111] text-green-400">
                <div className="prose prose-invert max-w-none text-sm leading-relaxed">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {stream + "▌"}
                  </ReactMarkdown>
                </div>
              </div>

            </div>
          </motion.div>
        )}

        {/* LOADING */}
        {!isStreaming && messages.length > 0 && (
          <div className="text-gray-500 text-sm animate-pulse">
            AI is thinking...
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
