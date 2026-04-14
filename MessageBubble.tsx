"use client";

import {
  User,
  Bot,
  Copy,
  ThumbsUp,
  ThumbsDown,
  RefreshCw,
  MoreHorizontal,
} from "lucide-react";
import { useState, useMemo } from "react";
import { useStore } from "../../store/useStore";

type Props = {
  id: string;
  role: "user" | "ai";
  content: string;
};

export default function MessageBubble({ id, role, content }: Props) {
  const [copied, setCopied] = useState(false);
  const [showThinking, setShowThinking] = useState(false);

  const {
    currentChatId,
    conversations,
    pinMessage,
    unpinMessage,
    toggleSelectMessage,
    selectedMessages,
    thinking, // 🔥 NEW
  } = useStore();

  const currentChat = useMemo(
    () =>
      conversations.find((c) => c.id === currentChatId) ?? null,
    [conversations, currentChatId]
  );

  const isPinned = (currentChat?.pinnedMessages || []).some(
    (m) => m.id === id
  );

  const isSelected = selectedMessages.some(
    (m) => m.id === id
  );

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const handleRegenerate = (text: string) => {
    console.log("Regenerate:", text);
  };

  return (
    <div
      className={`flex flex-col ${
        role === "user" ? "items-end" : "items-start"
      }`}
    >
      {/* MAIN ROW */}
      <div className="flex items-center gap-2">
        <input
          type="checkbox"
          checked={isSelected}
          onChange={() =>
            toggleSelectMessage({ id, role, content })
          }
        />

        {role === "ai" && (
          <div className="w-8 h-8 flex items-center justify-center rounded-full bg-white/10">
            <Bot size={16} />
          </div>
        )}

        <div
          className={`
            relative group px-4 py-3 rounded-2xl max-w-[70%] text-sm leading-relaxed shadow-md
            ${
              role === "user"
                ? "bg-blue-600 text-white"
                : "bg-white/5 backdrop-blur-md border border-white/10 text-gray-100"
            }
            ${isPinned ? "ring-2 ring-yellow-400/70" : ""}
          `}
        >
          <div className="whitespace-pre-wrap break-words">
            {content}
          </div>

          {role === "ai" && (
            <button
              onClick={handleCopy}
              className="absolute -top-2 -right-2 opacity-0 group-hover:opacity-100 
              bg-black/50 p-1 rounded transition"
            >
              <Copy size={14} />
            </button>
          )}

          <button
            onClick={() =>
              isPinned
                ? unpinMessage(currentChatId, { id, role, content })
                : pinMessage(currentChatId, { id, role, content })
            }
            className="absolute -bottom-2 right-2 opacity-0 group-hover:opacity-100 
            bg-black/50 px-2 py-0.5 rounded text-xs hover:bg-yellow-500/50 transition"
          >
            {isPinned ? "Unpin" : "Pin"}
          </button>

          {isPinned && (
            <div className="absolute -left-2 top-2 text-yellow-400 text-xs">
              📌
            </div>
          )}

          {copied && (
            <div className="absolute -top-6 right-0 text-xs text-green-400">
              Copied!
            </div>
          )}
        </div>

        {role === "user" && (
          <div className="w-8 h-8 flex items-center justify-center rounded-full bg-blue-600">
            <User size={16} />
          </div>
        )}
      </div>

      {/* 🔥 THINKING PANEL */}
      {role === "ai" && thinking && (
        <div className="ml-10 mt-2 max-w-[70%]">
          <div
            className="text-xs text-gray-400 cursor-pointer hover:text-white transition"
            onClick={() => setShowThinking(!showThinking)}
          >
            Show thinking {showThinking ? "▲" : "▼"}
          </div>

          {showThinking && (
            <div
              className="mt-2 bg-black/40 border border-white/10 rounded-lg p-3 
              text-xs text-gray-300 font-mono whitespace-pre-wrap"
            >
              {thinking}
            </div>
          )}
        </div>
      )}

      {/* ACTION BUTTONS */}
      {role === "ai" && (
        <div className="flex items-center gap-3 mt-1 ml-10 text-gray-400 text-xs">
          <button
            onClick={handleCopy}
            className="hover:text-white flex items-center gap-1"
          >
            <Copy size={14} /> Copy
          </button>

          <button className="hover:text-green-400">
            <ThumbsUp size={14} />
          </button>

          <button className="hover:text-red-400">
            <ThumbsDown size={14} />
          </button>

          <button
            onClick={() => handleRegenerate(content)}
            className="hover:text-blue-400"
          >
            <RefreshCw size={14} />
          </button>

          <button className="hover:text-white">
            <MoreHorizontal size={14} />
          </button>
        </div>
      )}
    </div>
  );
}