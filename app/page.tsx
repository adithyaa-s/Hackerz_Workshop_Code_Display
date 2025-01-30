"use client"

import { useEffect } from "react"
import CodeCard from "@/components/CodeCard"
import { motion, useAnimation } from "framer-motion"

export default function Home() {
  const controls = useAnimation()

  const codeSnippets = [
    {
      title: "Model Architecture",
      code: `class AIModel(nn.Module):
    def __init__(self):
        super(AIModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)`,
    },
    {
      title: "Training Loop",
      code: `for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}')`,
    },
    {
      title: "Evaluation Function",
      code: `def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')`,
    },
  ]

  useEffect(() => {
    controls.start((i) => ({
      opacity: 1,
      y: 0,
      transition: { delay: i * 0.2 },
    }))
  }, [controls])

  return (
    <div className="min-h-screen bg-background text-foreground">
      <main className="container mx-auto px-4 py-8">
        <motion.h1
          className="text-4xl font-bold text-neon-purple mb-8 text-center"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          AI Model Showcase
        </motion.h1>
        <div className="space-y-8">
          {codeSnippets.map((snippet, index) => (
            <motion.div key={snippet.title} initial={{ opacity: 0, y: 20 }} animate={controls} custom={index}>
              <CodeCard title={snippet.title} code={snippet.code} />
            </motion.div>
          ))}
        </div>
      </main>
    </div>
  )
}

