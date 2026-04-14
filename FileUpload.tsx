"use client";

/* ✅ Correct event type */
export default function FileUpload() {
  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    await fetch("http://localhost:8000/upload", {
      method: "POST",
      body: formData,
    });
  };

  return (
    <input
      type="file"
      onChange={handleUpload}
      className="text-sm"
    />
  );
}