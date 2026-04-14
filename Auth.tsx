"use client";

import { useStore } from "../store/useStore";

export default function Auth() {
  const { user, login, logout } = useStore();

  return user ? (
    <div className="flex gap-2 items-center">
      <span>{user}</span>
      <button onClick={logout} className="text-red-400">
        Logout
      </button>
    </div>
  ) : (
    <button
      onClick={() => login("demo@user.com")}
      className="bg-blue-500 px-4 py-2 rounded"
    >
      Login
    </button>
  );
}