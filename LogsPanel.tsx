"use client";

import { useStore } from "../store/useStore";

export default function LogsPanel() {
  const { logs } = useStore();

  return (
    <div className="bg-white/5 border border-white/10 rounded-xl p-3 h-[180px] overflow-y-auto text-xs space-y-1">
      {logs.map((log) => (
        <div key={log.id} className="flex justify-between">

          {/* 📝 MESSAGE */}
          <span
            className={
              log.type === "error"
                ? "text-red-400"
                : log.type === "warn"
                ? "text-yellow-400"
                : "text-gray-300"
            }
          >
            • {log.message}
          </span>

          {/* ⏱ TIME */}
          <span className="text-gray-500 text-[10px] ml-2">
            {new Date(log.time).toLocaleTimeString()}
          </span>

        </div>
      ))}
    </div>
  );
}