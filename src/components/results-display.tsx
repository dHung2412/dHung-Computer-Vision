"use client"

import { Card } from "@/components/ui/card"

interface ResultsDisplayProps {
  results: {
    svm?: string
    rf?: string
  }
}

export default function ResultsDisplay({ results }: ResultsDisplayProps) {
  return (
    <div className="space-y-6">
      {results.svm && (
        <Card className="overflow-hidden">
          <div className="border-b border-border bg-muted/50 px-6 py-4">
            <h3 className="font-semibold text-foreground">SVM Detection</h3>
            <p className="text-xs text-muted-foreground">Support Vector Machine results</p>
          </div>
          <div className="p-4">
            <div className="relative aspect-video overflow-hidden rounded-lg border border-border bg-black">
              <img src={results.svm || "/placeholder.svg"} alt="SVM Result" className="h-full w-full object-contain" />
            </div>
          </div>
        </Card>
      )}

      {results.rf && (
        <Card className="overflow-hidden">
          <div className="border-b border-border bg-muted/50 px-6 py-4">
            <h3 className="font-semibold text-foreground">Random Forest Detection</h3>
            <p className="text-xs text-muted-foreground">Random Forest model results</p>
          </div>
          <div className="p-4">
            <div className="relative aspect-video overflow-hidden rounded-lg border border-border bg-black">
              <img src={results.rf || "/placeholder.svg"} alt="RF Result" className="h-full w-full object-contain" />
            </div>
          </div>
        </Card>
      )}
    </div>
  )
}
