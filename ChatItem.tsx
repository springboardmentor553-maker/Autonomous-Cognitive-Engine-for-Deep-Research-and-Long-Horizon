"use client";

import { useState, useRef, useEffect } from "react";
import { useStore } from "../../store/useStore";
import { MessageSquare } from "lucide-react";

type Props = {
  chat: {
    id: string;
    title: string;
  };
};

export default function ChatItem({ chat }: Props) {
  const {
    setCurrentChat,
    currentChatId,
    renameChat,
    conversations,
    folders,
    assignFolder,
  } = useStore();

  const [menuOpen, setMenuOpen] = useState(false);
  const [menuPos, setMenuPos] = useState({ x: 0, y: 0 });
  const [editing, setEditing] = useState(false);
  const [newTitle, setNewTitle] = useState(chat.title);

  const menuRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const isActive = currentChatId === chat.id;

  /* =========================
     CLICK OUTSIDE CLOSE
  ========================= */
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    };

    if (menuOpen) {
      window.addEventListener("click", handler);
    }

    return () => window.removeEventListener("click", handler);
  }, [menuOpen]);

  /* =========================
     AUTO FOCUS INPUT
  ========================= */
  useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus();
    }
  }, [editing]);

  /* =========================
     RIGHT CLICK MENU
  ========================= */
  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();

    const x = Math.min(e.clientX, window.innerWidth - 180);
    const y = Math.min(e.clientY, window.innerHeight - 200);

    setMenuPos({ x, y });
    setMenuOpen(true);
  };

  /* =========================
     DELETE
  ========================= */
  const handleDelete = () => {
    const updated = conversations.filter((c) => c.id !== chat.id);

    useStore.setState({
      conversations: updated,
      currentChatId: updated[0]?.id || "",
    });

    setMenuOpen(false);
  };

  /* =========================
     INLINE RENAME
  ========================= */
  const handleRename = () => {
    setEditing(true);
    setMenuOpen(false);
  };

  const handleRenameSubmit = () => {
    if (newTitle.trim()) {
      renameChat(chat.id, newTitle.trim());
    }
    setEditing(false);
  };

  /* =========================
     MOVE
  ========================= */
  const handleMove = (folderId: string) => {
    assignFolder(chat.id, folderId);
    setMenuOpen(false);
  };

  return (
    <>
      {/* CHAT ITEM */}
      <div
        draggable
        onDragStart={(e) => {
          e.dataTransfer.setData("chatId", chat.id);
        }}
        onClick={() => setCurrentChat(chat.id)}
        onContextMenu={handleContextMenu}
        className={`flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer text-sm transition
        ${
          isActive
            ? "bg-blue-600/30 border border-blue-500/30"
            : "hover:bg-white/10"
        }`}
      >
        <MessageSquare size={16} className="opacity-70" />

        {/* 🔥 INLINE EDIT */}
        {editing ? (
          <input
            ref={inputRef}
            value={newTitle}
            onChange={(e) => setNewTitle(e.target.value)}
            onBlur={handleRenameSubmit}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleRenameSubmit();
            }}
            className="bg-transparent outline-none text-sm flex-1"
          />
        ) : (
          <span className="truncate flex-1">{chat.title}</span>
        )}
      </div>

      {/* CONTEXT MENU */}
      {menuOpen && (
        <div
          ref={menuRef}
          className="fixed z-50 bg-[#1e1e1e] border border-white/10 rounded-lg shadow-lg p-2 text-sm"
          style={{ top: menuPos.y, left: menuPos.x }}
        >
          <button
            onClick={handleRename}
            className="block w-full text-left px-3 py-1 hover:bg-white/10 rounded"
          >
            ✏️ Rename
          </button>

          <button
            onClick={handleDelete}
            className="block w-full text-left px-3 py-1 hover:bg-red-500/20 text-red-400 rounded"
          >
            🗑 Delete
          </button>

          <div className="mt-1 border-t border-white/10 pt-1">
            <div className="text-xs text-gray-400 px-2 mb-1">
              Move to folder
            </div>

            {folders.length === 0 ? (
              <div className="text-xs text-gray-500 px-2">
                No folders
              </div>
            ) : (
              folders.map((f) => (
                <button
                  key={f.id}
                  onClick={() => handleMove(f.id)}
                  className="block w-full text-left px-3 py-1 hover:bg-white/10 rounded"
                >
                  📁 {f.name}
                </button>
              ))
            )}
          </div>
        </div>
      )}
    </>
  );
}