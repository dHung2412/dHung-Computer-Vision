"use client"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import DetectionForm from "@/components/detection-form"
import ResultsDisplay from "@/components/results-display"

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string>("")
  const [results, setResults] = useState<{ svm?: string; rf?: string } | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>("")
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file")
      return
    }

    setSelectedFile(file)
    setError("")

    // Create preview
    const reader = new FileReader()
    reader.onload = (e) => {
      setPreview(e.target?.result as string)
    }
    reader.readAsDataURL(file)
  }

  const handleDetect = async (model: "svm" | "rf" | "both") => {
    if (!selectedFile) {
      setError("Please select an image first")
      return
    }

    setLoading(true)
    setError("")
    setResults(null)

    try {
      const formData = new FormData()
      formData.append("image", selectedFile)

      let response

      if (model === "both") {
        response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:5000"}/detect/both`, {
          method: "POST",
          body: formData,
        })

        if (!response.ok) {
          throw new Error("Detection failed")
        }

        const data = await response.json()

        const svmImg = data.svm?.image ? `data:image/jpeg;base64,${data.svm.image}` : null
        const rfImg = data.rf?.image ? `data:image/jpeg;base64,${data.rf.image}` : null

        setResults({
          svm: svmImg || undefined,
          rf: rfImg || undefined,
        })
      } else {
        const endpoint = model === "svm" ? "/detect/svm" : "/detect/rf"
        response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:5000"}${endpoint}`, {
          method: "POST",
          body: formData,
        })

        if (!response.ok) {
          throw new Error("Detection failed")
        }

        const blob = await response.blob()
        const imgUrl = URL.createObjectURL(blob)

        setResults({
          [model]: imgUrl,
        })
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred during detection")
      setResults(null)
    } finally {
      setLoading(false)
    }
  }

  const handleClear = () => {
    setSelectedFile(null)
    setPreview("")
    setResults(null)
    setError("")
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  return (
    <main className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card">
        <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6 lg:px-8">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary">
              <svg className="h-6 w-6 text-primary-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <div>
              <h1 className="text-2xl font-bold text-foreground">Traffic Sign Detector</h1>
              <p className="text-sm text-muted-foreground">AI-powered traffic sign recognition</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="mx-auto max-w-6xl px-4 py-12 sm:px-6 lg:px-8">
        <div className="grid gap-8 lg:grid-cols-2">
          {/* Left: Upload & Controls */}
          <div className="space-y-6">
            <Card className="p-8">
              <h2 className="mb-6 text-xl font-semibold text-foreground">Upload Image</h2>

              {!preview ? (
                <DetectionForm fileInputRef={fileInputRef} onFileSelect={handleFileSelect} />
              ) : (
                <div className="space-y-4">
                  <div className="relative aspect-video overflow-hidden rounded-lg border border-border bg-muted">
                    <img src={preview || "/placeholder.svg"} alt="Preview" className="h-full w-full object-contain" />
                  </div>

                  <div className="flex gap-2">
                    <Button variant="outline" size="sm" onClick={() => fileInputRef.current?.click()}>
                      Change Image
                    </Button>
                    <Button variant="outline" size="sm" onClick={handleClear}>
                      Clear
                    </Button>
                  </div>
                </div>
              )}
            </Card>

            {/* Model Selection */}
            {preview && (
              <Card className="p-8">
                <h3 className="mb-6 text-lg font-semibold text-foreground">Detection Model</h3>

                <div className="space-y-3">
                  <Button onClick={() => handleDetect("svm")} disabled={loading} className="w-full">
                    {loading ? "Processing..." : "Detect with SVM"}
                  </Button>

                  <Button onClick={() => handleDetect("rf")} disabled={loading} variant="secondary" className="w-full">
                    {loading ? "Processing..." : "Detect with Random Forest"}
                  </Button>

                  <Button onClick={() => handleDetect("both")} disabled={loading} variant="outline" className="w-full">
                    {loading ? "Processing..." : "Detect with Both Models"}
                  </Button>
                </div>

                <p className="mt-4 text-xs text-muted-foreground">
                  Compare different ML models for traffic sign detection. Both models process your image and return
                  annotated results with confidence scores.
                </p>
              </Card>
            )}
          </div>

          {/* Right: Results */}
          <div>
            {error && (
              <Card className="border-destructive bg-destructive/10 p-6">
                <p className="text-sm text-destructive">{error}</p>
              </Card>
            )}

            {loading && (
              <Card className="flex items-center justify-center p-12">
                <div className="text-center">
                  <div className="mb-4 h-12 w-12 animate-spin rounded-full border-4 border-border border-t-primary mx-auto" />
                  <p className="text-muted-foreground">Processing image...</p>
                </div>
              </Card>
            )}

            {results && !loading && <ResultsDisplay results={results} />}

            {!results && !loading && !error && preview && (
              <Card className="flex items-center justify-center p-12">
                <p className="text-center text-muted-foreground">Select a detection model to analyze the image</p>
              </Card>
            )}
          </div>
        </div>

        {/* Footer Info */}
        {!preview && (
          <Card className="mt-12 bg-card/50 p-8 backdrop-blur">
            <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
              <div>
                <h4 className="font-semibold text-foreground">Supported Signs</h4>
                <p className="mt-2 text-sm text-muted-foreground">
                  Detects Vietnamese traffic signs including stop signs, prohibition signs, warning signs, and more.
                </p>
              </div>
              <div>
                <h4 className="font-semibold text-foreground">Two Models</h4>
                <p className="mt-2 text-sm text-muted-foreground">
                  Compare SVM and Random Forest models to find which works best for your use case.
                </p>
              </div>
              <div>
                <h4 className="font-semibold text-foreground">Confidence Scores</h4>
                <p className="mt-2 text-sm text-muted-foreground">
                  Each detection includes a confidence score showing how certain the model is about the prediction.
                </p>
              </div>
            </div>
          </Card>
        )}
      </div>

      <input
        type="file"
        ref={fileInputRef}
        onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
        accept="image/*"
        className="hidden"
      />
    </main>
  )
}
