"use client";

import { create } from "zustand";

/* =========================
   TYPES
========================= */
export type Message = {
  id: string;
  role: "user" | "ai";
  content: string;
};

type Folder = {
  id: string;
  name: string;
};

type Conversation = {
  id: string;
  title: string;
  messages: Message[];
  folderId?: string;
  pinned?: boolean;
  pinnedMessages?: Message[];
  createdAt?: number;
};

type Log = {
  id: string;
  message: string;
  type: "info" | "error" | "warn";
  time: number;
};

type State = {
  query: string;
  stream: string;

  logs: Log[];
  history: string[];

  conversations: Conversation[];
  currentChatId: string;

  folders: Folder[];
  search: string;

  selectedMessages: Message[];
  error: string | null;

  sidebarOpen: boolean;
  thinking: string;

  setQuery: (q: string) => void;
  appendStream: (chunk: string) => void;
  clearStream: () => void;

  addLog: (msg: string, type?: Log["type"]) => void;
  saveHistory: (entry: string) => void;

  setCurrentChat: (id: string) => void;
  addConversation: (conv: Conversation) => void;
  createNewChat: () => void;

  addMessageToCurrent: (msg: Message) => void;
  replaceLastMessage: (text: string) => void;

  clearMessages: () => void;
  setError: (err: string | null) => void;
  setSearch: (s: string) => void;

  addFolder: (name: string) => void;
  assignFolder: (chatId: string, folderId: string) => void;

  togglePin: (chatId: string) => void;
  deleteChat: (chatId: string) => void;
  renameChat: (chatId: string, newTitle: string) => void;

  pinMessage: (chatId: string, msg: Message) => void;
  unpinMessage: (chatId: string, msg: Message) => void;
  toggleSelectMessage: (msg: Message) => void;

  toggleSidebar: () => void;
  setSidebar: (v: boolean) => void;

  setThinking: (text: string) => void;
};

/* =========================
   HELPERS
========================= */
const uid = () =>
  typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : Math.random().toString(36).substring(2);

const load = () => {
  if (typeof window === "undefined") return null;
  try {
    return JSON.parse(localStorage.getItem("chat-storage") || "null");
  } catch {
    return null;
  }
};

const save = (state: any) => {
  if (typeof window !== "undefined") {
    localStorage.setItem(
      "chat-storage",
      JSON.stringify({
        conversations: state.conversations,
        currentChatId: state.currentChatId,
        folders: state.folders,
      })
    );
  }
};

const saved = load() || {
  conversations: [],
  currentChatId: "",
  folders: [],
};

/* =========================
   STORE
========================= */
export const useStore = create<State>()((set, get) => ({
  query: "",
  stream: "",

  logs: [],
  history: [],

  conversations: Array.isArray(saved.conversations)
    ? saved.conversations
    : [],
  currentChatId: saved.currentChatId || "",

  folders: saved.folders || [],
  search: "",

  selectedMessages: [],
  error: null,

  sidebarOpen: true,
  thinking: "",

  toggleSidebar: () => {
    const next = !get().sidebarOpen;
    localStorage.setItem("sidebar-open", String(next));
    set({ sidebarOpen: next });
  },

  setSidebar: (v) => {
    localStorage.setItem("sidebar-open", String(v));
    set({ sidebarOpen: v });
  },

  appendStream: (chunk) =>
    set((s) => ({
      stream: s.stream + chunk,
    })),

  clearStream: () => set({ stream: "" }),

  addLog: (message, type = "info") =>
    set((s) => ({
      logs: [
        ...s.logs,
        {
          id: uid(),
          message,
          type,
          time: Date.now(),
        },
      ],
    })),

  setQuery: (q) => set({ query: q }),

  saveHistory: (entry) =>
    set((s) => ({ history: [entry, ...s.history] })),

  setCurrentChat: (id) => set({ currentChatId: id, stream: "" }),

  createNewChat: () => {
    const id = uid();

    const newChat: Conversation = {
      id,
      title: "New Chat",
      messages: [],
      pinnedMessages: [],
      createdAt: Date.now(),
      pinned: false,
    };

    set((s) => {
      const updated = [newChat, ...s.conversations];

      const newState = {
        conversations: updated,
        currentChatId: id,
      };

      save({ ...s, ...newState });
      return newState;
    });
  },

  addConversation: (conv) =>
    set((s) => {
      const updated = [
        {
          ...conv,
          pinnedMessages: conv.pinnedMessages || [],
          createdAt: Date.now(),
        },
        ...s.conversations,
      ];

      const newState = {
        conversations: updated,
        currentChatId: conv.id,
      };

      save({ ...s, ...newState });
      return newState;
    }),

  /* 🔥 FIXED HERE */
  addMessageToCurrent: (msg) =>
    set((s) => {
      const updated = s.conversations.map((c) => {
        if (c.id !== s.currentChatId) return c;

        const safeMessages = Array.isArray(c.messages)
          ? c.messages
          : [];

        const isFirstMessage = safeMessages.length === 0;

        return {
          ...c,
          title: isFirstMessage
            ? msg.content.split(" ").slice(0, 6).join(" ")
            : c.title,
          messages: [
            ...safeMessages,
            {
              id: msg.id || uid(),
              role: msg.role as "user" | "ai", // 🔥 FIX
              content: msg.content,
            },
          ],
        };
      });

      save({ ...s, conversations: updated });
      return { conversations: updated };
    }),

  replaceLastMessage: (text) =>
    set((s) => {
      const updated = s.conversations.map((c) => {
        if (c.id !== s.currentChatId) return c;

        const safeMessages = Array.isArray(c.messages)
          ? c.messages
          : [];

        return {
          ...c,
          messages: [
            ...safeMessages.slice(0, -1),
            {
              id: uid(),
              role: "ai" as const,
              content: text,
            },
          ],
        };
      });

      save({ ...s, conversations: updated });
      return { conversations: updated };
    }),

  renameChat: (chatId, newTitle) =>
    set((s) => {
      const updated = s.conversations.map((c) =>
        c.id === chatId ? { ...c, title: newTitle } : c
      );

      save({ ...s, conversations: updated });
      return { conversations: updated };
    }),

  pinMessage: (chatId, msg) =>
    set((s) => ({
      conversations: s.conversations.map((c) =>
        c.id === chatId
          ? {
              ...c,
              pinnedMessages: [...(c.pinnedMessages || []), msg],
            }
          : c
      ),
    })),

  unpinMessage: (chatId, msg) =>
    set((s) => ({
      conversations: s.conversations.map((c) =>
        c.id === chatId
          ? {
              ...c,
              pinnedMessages: (c.pinnedMessages || []).filter(
                (m) => m.id !== msg.id
              ),
            }
          : c
      ),
    })),

  toggleSelectMessage: (msg) =>
    set((s) => {
      const exists = s.selectedMessages.some((m) => m.id === msg.id);

      return {
        selectedMessages: exists
          ? s.selectedMessages.filter((m) => m.id !== msg.id)
          : [...s.selectedMessages, msg],
      };
    }),

  clearMessages: () =>
    set({
      conversations: [],
      currentChatId: "",
    }),

  setError: (err) => set({ error: err }),
  setSearch: (s) => set({ search: s }),

  addFolder: (name) =>
    set((s) => ({
      folders: [...s.folders, { id: uid(), name }],
    })),

  assignFolder: (chatId, folderId) =>
    set((s) => ({
      conversations: s.conversations.map((c) =>
        c.id === chatId ? { ...c, folderId } : c
      ),
    })),

  togglePin: (chatId) =>
    set((s) => ({
      conversations: s.conversations.map((c) =>
        c.id === chatId ? { ...c, pinned: !c.pinned } : c
      ),
    })),

  deleteChat: (chatId) =>
    set((s) => {
      const updated = s.conversations.filter((c) => c.id !== chatId);

      save({ ...s, conversations: updated });
      return { conversations: updated };
    }),

  setThinking: (text) => set({ thinking: text }),
}));