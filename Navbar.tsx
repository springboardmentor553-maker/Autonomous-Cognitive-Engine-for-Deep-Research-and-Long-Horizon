
"use client";

import { useEffect, useState } from "react";
import { LogOut, User, Menu } from "lucide-react";
import { useStore } from "../../store/useStore";

export default function Navbar() {
  const [email, setEmail] = useState("User");

  const { toggleSidebar } = useStore(); // 🔥 ADD

  useEffect(() => {
    const token = localStorage.getItem("token");

    if (!token) return;

    try {
      const payload = JSON.parse(atob(token.split(".")[1]));
      setEmail(payload.email || "User");
    } catch {
      console.error("Invalid token");
    }
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("token");
    window.location.href = "/login";
  };

  return (
    <div className="h-14 px-4 flex items-center justify-between 
    bg-white/5 backdrop-blur-md border-b border-white/10 shadow-sm">

      {/* LEFT */}
      <div className="flex items-center gap-3">

        {/* 🔥 SIDEBAR TOGGLE */}
        <button
          onClick={toggleSidebar}
          className="p-2 rounded hover:bg-white/10 transition"
        >
          <Menu size={20} />
        </button>

        {/* TITLE */}
        <div className="font-semibold text-lg tracking-wide">
          🤖 AI Engine
        </div>
      </div>

      {/* RIGHT */}
      <div className="flex items-center gap-4">

        {/* USER */}
        <div className="flex items-center gap-2 text-sm text-gray-300 bg-white/5 px-3 py-1 rounded-lg border border-white/10">
          <User size={16} />
          <span className="truncate max-w-[150px]">
            {email}
          </span>
        </div>

        {/* LOGOUT */}
        <button
          onClick={handleLogout}
          className="flex items-center gap-2 bg-red-600/80 hover:bg-red-600 px-3 py-1 rounded-lg text-sm transition-all duration-200"
        >
          <LogOut size={16} />
          Logout
        </button>

      </div>
    </div>
  );
}
