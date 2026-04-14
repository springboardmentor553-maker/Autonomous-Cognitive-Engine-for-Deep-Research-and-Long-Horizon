"use client";

import ReactFlow from "reactflow";
import "reactflow/dist/style.css";

export default function AgentFlow() {
  const nodes = [
    { id: "1", data: { label: "Planner" }, position: { x: 0, y: 0 } },
    { id: "2", data: { label: "Research" }, position: { x: 200, y: 100 } },
    { id: "3", data: { label: "Analysis" }, position: { x: 400, y: 0 } },
    { id: "4", data: { label: "Summary" }, position: { x: 600, y: 100 } },
  ];

  const edges = [
    { id: "e1-2", source: "1", target: "2" },
    { id: "e2-3", source: "2", target: "3" },
    { id: "e3-4", source: "3", target: "4" },
  ];

  return (
    <div className="h-[300px]">
      <ReactFlow nodes={nodes} edges={edges} fitView />
    </div>
  );
}