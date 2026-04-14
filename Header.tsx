import ExportPDF from "../ExportPDF";

export default function Header() {
  return (
    <div className="px-6 py-4 border-b border-white/10 flex justify-between">
      <span className="font-semibold">AI Assistant</span>
      <ExportPDF />
    </div>
  );
}