"use client";

export default function TypingLoader() {
  return (
    <div className="flex justify-start">
      <div className="bg-[#1e1e1e] p-4 rounded-2xl max-w-lg w-full space-y-2 animate-pulse">

        <div className="h-3 bg-gray-600 rounded w-3/4"></div>
        <div className="h-3 bg-gray-600 rounded w-1/2"></div>
        <div className="h-3 bg-gray-600 rounded w-2/3"></div>

      </div>
    </div>
  );
}