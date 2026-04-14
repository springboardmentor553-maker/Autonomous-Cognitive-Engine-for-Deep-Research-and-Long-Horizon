"use client";

import jsPDF from "jspdf";
import { useStore } from "../store/useStore";

export default function ExportPDF() {
  const { stream } = useStore();

  const exportPDF = () => {
    const doc = new jsPDF();
    doc.text(stream, 10, 10);
    doc.save("research.pdf");
  };

  return (
    <button
      onClick={exportPDF}
      className="bg-green-600 px-4 py-2 rounded"
    >
      Export PDF
    </button>
  );
}