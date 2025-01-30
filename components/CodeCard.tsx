"use client"

import { useState } from "react"
import { ClipboardIcon, CheckIcon } from "lucide-react"
import { motion } from "framer-motion"
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter"
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism"

interface CodeCardProps {
  title: string
  code: string
}

export default function CodeCard({ title, code }: CodeCardProps) {
  const [copied, setCopied] = useState(false)

  const copyToClipboard = () => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <motion.div
      className="bg-card rounded-lg shadow-lg overflow-hidden"
      whileHover={{ scale: 1.02 }}
      transition={{ duration: 0.2 }}
    >
      <div className="flex justify-between items-center p-4 bg-neon-purple text-foreground">
        <h2 className="text-xl font-semibold">{title}</h2>
        <button
          onClick={copyToClipboard}
          className="text-foreground hover:text-neon-pink transition-colors duration-300"
        >
          {copied ? <CheckIcon className="w-5 h-5" /> : <ClipboardIcon className="w-5 h-5" />}
        </button>
      </div>
      <div className="p-4 bg-card">
        <SyntaxHighlighter
          language="python"
          style={vscDarkPlus}
          customStyle={{ margin: 0, borderRadius: "0.5rem", background: "#1E1E1E" }}
        >
          {code}
        </SyntaxHighlighter>
      </div>
    </motion.div>
  )
}

