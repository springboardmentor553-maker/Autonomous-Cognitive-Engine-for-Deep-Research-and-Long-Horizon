"use client";
import { useStore } from "../store/useStore";

export default function HistoryPanel() {
  const { history } = useStore();

  return (
    <div className="space-y-2">
      {history.map((item, i) => (
        <div key={i} className="text-xs bg-white/5 p-2 rounded">
          {item}
        </div>
      ))}
    </div>
  );
}