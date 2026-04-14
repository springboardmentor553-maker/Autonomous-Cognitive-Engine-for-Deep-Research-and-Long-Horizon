"use client";

import { useStore } from "../store/useStore";
import { useEffect, useRef } from "react";

export default function TypingOutput() {
  const { stream } = useStore();
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    ref.current?.scrollIntoView({ behavior: "smooth" });
  }, [stream]);

  if (!stream) return null;

  return (
    <div className="flex justify-start">
      <div
        ref={ref}
        className="bg-[#1e1e1e] px-4 py-3 rounded-xl max-w-2xl text-sm"
        style={{ whiteSpace: "pre-wrap" }}
      >
        {stream}
      </div>
    </div>
  );
}