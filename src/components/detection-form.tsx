"use client"

import type React from "react"

import { useRef } from "react"
import { Button } from "@/components/ui/button"

interface DetectionFormProps {
  fileInputRef: React.RefObject<HTMLInputElement>
  onFileSelect: (file: File) => void
}

export default function DetectionForm({ fileInputRef, onFileSelect }: DetectionFormProps) {
  const dragRef = useRef<HTMLDivElement>(null)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (dragRef.current) {
      dragRef.current.classList.add("border-primary", "bg-primary/5")
    }
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (dragRef.current) {
      dragRef.current.classList.remove("border-primary", "bg-primary/5")
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (dragRef.current) {
      dragRef.current.classList.remove("border-primary", "bg-primary/5")
    }

    const files = e.dataTransfer.files
    if (files && files[0]) {
      onFileSelect(files[0])
    }
  }

  return (
    <div
      ref={dragRef}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className="rounded-lg border-2 border-dashed border-border bg-muted/30 p-12 text-center transition-all"
    >
      <svg className="mx-auto h-12 w-12 text-muted-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6 20.25h12A2.25 2.25 0 0020.25 18V6A2.25 2.25 0 0018 3.75H6A2.25 2.25 0 003.75 6v12A2.25 2.25 0 006 20.25z"
        />
      </svg>

      <h3 className="mt-4 text-lg font-semibold text-foreground">Drag and drop your image</h3>
      <p className="mt-2 text-sm text-muted-foreground">or</p>

      <Button onClick={() => fileInputRef.current?.click()} className="mt-4">
        Browse Files
      </Button>

      <p className="mt-4 text-xs text-muted-foreground">PNG, JPG, GIF up to 10MB</p>
    </div>
  )
}
