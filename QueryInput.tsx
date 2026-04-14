"use client";

import { useState, useRef } from "react";
import { useStore } from "../store/useStore";
import { useStream } from "../hooks/useStream"; // 🔥 NEW
import { Send, Mic, Upload, X } from "lucide-react";

export default function QueryInput() {
  const {
    query,
    setQuery,
    addMessageToCurrent,
    setError,
  } = useStore();

  const { sendQuery } = useStream(); // 🔥 STREAM HOOK

  const [loading, setLoading] = useState(false);
  const [listening, setListening] = useState(false);
  const [files, setFiles] = useState<File[]>([]);
  const [dragActive, setDragActive] = useState(false);

  const recognitionRef = useRef<any>(null);

  /* 🎤 VOICE INPUT */
  const handleVoice = () => {
    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setError("Voice not supported");
      return;
    }

    if (!recognitionRef.current) {
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.lang = "en-US";

      recognitionRef.current.onresult = (e: any) => {
        setQuery(e.results[0][0].transcript);
      };

      recognitionRef.current.onerror = () => {
        setError("Voice recognition error");
        setListening(false);
      };

      recognitionRef.current.onend = () => setListening(false);
    }

    listening
      ? recognitionRef.current.stop()
      : recognitionRef.current.start();

    setListening(!listening);
  };

  /* 📂 FILE HANDLING */
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);

    const droppedFiles = Array.from(e.dataTransfer.files);

    const uniqueFiles = droppedFiles.filter(
      (f) => !files.some((existing) => existing.name === f.name)
    );

    setFiles((prev) => [...prev, ...uniqueFiles]);
  };

  const removeFile = (name: string) => {
    setFiles((prev) => prev.filter((f) => f.name !== name));
  };

  /* 🚀 SEND QUERY (🔥 FIXED VERSION) */
  const handleSubmit = async () => {
    if (!query.trim() || loading) return;

    const trimmed = query.trim();

    // 👤 USER MESSAGE
    addMessageToCurrent({
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
    });

    setLoading(true);
    setError(null);

    try {
      // 🔥 ONLY ONE STREAM SYSTEM
      await sendQuery(trimmed);
    } catch {
      console.log("Error sending query");
    } finally {
      setLoading(false);
      setQuery("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") handleSubmit();
  };

  return (
    <div
      className={`flex flex-col gap-2 p-3 rounded-2xl border transition-all
      ${
        dragActive
          ? "border-blue-500 bg-blue-500/10 scale-[1.01]"
          : "border-white/10 bg-white/5"
      }`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragActive(true);
      }}
      onDragLeave={() => setDragActive(false)}
      onDrop={handleDrop}
    >
      {files.length > 0 && (
        <div className="flex flex-wrap gap-2 text-xs">
          {files.map((file) => (
            <div
              key={file.name}
              className="flex items-center gap-1 bg-green-500/10 px-2 py-1 rounded"
            >
              📎 {file.name}
              <button onClick={() => removeFile(file.name)}>
                <X size={12} />
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="flex items-center gap-2">
        <label className="p-2 bg-white/10 rounded-lg cursor-pointer">
          <Upload size={16} />
          <input
            type="file"
            multiple
            hidden
            onChange={(e) => {
              const selected = e.target.files
                ? Array.from(e.target.files)
                : [];
              setFiles((prev) => [...prev, ...selected]);
            }}
          />
        </label>

        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          className="flex-1 bg-transparent outline-none text-sm"
          placeholder="Ask anything..."
        />

        <button
          onClick={handleVoice}
          className={`p-2 rounded-lg ${
            listening ? "bg-red-500" : "bg-white/10"
          }`}
        >
          <Mic size={16} />
        </button>

        <button
          onClick={handleSubmit}
          disabled={loading}
          className={`px-3 py-1 rounded-lg ${
            loading
              ? "bg-blue-600/50 cursor-not-allowed"
              : "bg-blue-600 hover:bg-blue-700"
          }`}
        >
          {loading ? "..." : <Send size={16} />}
        </button>
      </div>
    </div>
  );
}