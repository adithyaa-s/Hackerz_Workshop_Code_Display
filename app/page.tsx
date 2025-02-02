"use client"

import { useEffect } from "react"
import CodeCard from "@/components/CodeCard"
import { motion, useAnimation } from "framer-motion"

export default function Home() {
  const controls = useAnimation()
  const codeSnippets = [
    {
      file: "Nltkutils.py",
      snippets: [
        {
          title: "Library Imports and Initialization",
          code: `import numpy as np
  import nltk
  nltk.download('punkt')  
  from nltk.stem.porter import PorterStemmer
  stemmer = PorterStemmer()  
  `,
        },
        {
          title: "Tokenization Function",
          code: `def tokenize(sentence):
      """
      Split sentence into an array of words/tokens.
      A token can be a word, punctuation, or number.
      """
      return nltk.word_tokenize(sentence)
  `,
        },
        {
          title: "Stemming Function",
          code: `def stem(word):
      """
      Perform stemming to find the root form of a word.
      """
      return stemmer.stem(word.lower())
  `,
        },
        {
          title: "Bag of Words Function",
          code: `def bag_of_words(tokenized_sentence, words):
      """
      Return a bag-of-words array.
      """
      sentence_words = [stem(word) for word in tokenized_sentence]
      bag = np.zeros(len(words), dtype=np.float32)
      for idx, w in enumerate(words):
          if w in sentence_words: 
              bag[idx] = 1
      return bag
  `,
        },
      ],
    },
    {
      file: "Model.py",
      snippets: [
        {
          title: "Importing PyTorch and Defining the Model",
          code: `import torch
  import torch.nn as nn
  `,
        },
        {
          title: "Defining the Neural Network Class",
          code: `class NeuralNet(nn.Module):
      def __init__(self, input_size, hidden_size, num_classes):
          super(NeuralNet, self).__init__()
          self.l1 = nn.Linear(input_size, hidden_size) 
          self.l2 = nn.Linear(hidden_size, hidden_size) 
          self.l3 = nn.Linear(hidden_size, num_classes)
          self.relu = nn.ReLU()
  `,
        },
        {
          title: "Forward Propagation Method",
          code: `def forward(self, x):
      out = self.l1(x)
      out = self.relu(out)
      out = self.l2(out)
      out = self.relu(out)
      out = self.l3(out)
      return out
  `,
        },
      ],
    },
    {
      file: "Train.py",
      snippets: [
        {
          title: "Importing Dependencies and Loading Data",
          code: `import numpy as np
  import random
  import json
  import nltk
  import torch
  import torch.nn as nn
  from torch.utils.data import Dataset, DataLoader
  from nltk_utils import bag_of_words, tokenize, stem
  from model import NeuralNet
  nltk.download('punkt')  
  with open('intents.json', 'r') as f:
      intents = json.load(f)
  `,
        },
        {
          title: "Preprocessing Data",
          code: `all_words = []
  tags = []
  xy = []
  for intent in intents['intents']:
      tag = intent['tag']
      tags.append(tag)
      for pattern in intent['patterns']:
          w = tokenize(pattern)
          all_words.extend(w)
          xy.append((w, tag))
  
  ignore_words = ['?', '.', '!']
  all_words = [stem(w) for w in all_words if w not in ignore_words]
  all_words = sorted(set(all_words))
  tags = sorted(set(tags))
  `,
        },
        {
          title: "Preparing Training Data",
          code: `X_train = []
  y_train = []
  for (pattern_sentence, tag) in xy:
      bag = bag_of_words(pattern_sentence, all_words)
      X_train.append(bag)
      label = tags.index(tag)
      y_train.append(label)
  
  X_train = np.array(X_train)
  y_train = np.array(y_train)
  `,
        },
        {
          title: "Defining Dataset and DataLoader",
          code: `class ChatDataset(Dataset):
      def __init__(self):
          self.n_samples = len(X_train)
          self.x_data = X_train
          self.y_data = y_train
  
      def __getitem__(self, index):
          return self.x_data[index], self.y_data[index]
  
      def __len__(self):
          return self.n_samples
  
  dataset = ChatDataset()
  train_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)
  `,
        },
        {
          title: "Training Loop",
          code: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = NeuralNet(input_size, hidden_size, output_size).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  
  for epoch in range(1000):
      for words, labels in train_loader:
          words = words.to(device)
          labels = labels.to(dtype=torch.long).to(device)
          
          outputs = model(words)
          loss = criterion(outputs, labels)
          
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
  
      if (epoch + 1) % 100 == 0:
          print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}')
  `,
        },
        {
          title: "Saving the Model",
          code: `data = {
      "model_state": model.state_dict(),
      "input_size": input_size,
      "hidden_size": hidden_size,
      "output_size": output_size,
      "all_words": all_words,
      "tags": tags
  }
  
  FILE = "data.pth"
  torch.save(data, FILE)
  print(f'Training complete. File saved to {FILE}')
  `,
        },
      ],
    },
    {
      file: "Chatbot/app.py",
      snippets: [
        {
          title: "Import Statements and Flask App Setup",
          code: `from flask import Flask, render_template, request, jsonify
  import torch
  import random
  import json
  from model import NeuralNet
  from nltk_utils import bag_of_words, tokenize
  
  app = Flask(__name__)
  `,
        },
        {
          title: "Loading Intents and Model Data",
          code: `with open('intents.json', 'r') as json_data:
      intents = json.load(json_data)
  
  FILE = "data.pth"
  data = torch.load(FILE)
  
  input_size = data["input_size"]
  hidden_size = data["hidden_size"]
  output_size = data["output_size"]
  all_words = data['all_words']
  tags = data['tags']
  model_state = data["model_state"]
  `,
        },
        {
          title: "Initializing the Neural Network Model",
          code: `model = NeuralNet(input_size, hidden_size, output_size)
  model.load_state_dict(model_state)
  model.eval()
  `,
        },
        {
          title: "Processing the User Input and Generating a Response",
          code: `def process_input(input_text):
      sentence = tokenize(input_text)
      X = bag_of_words(sentence, all_words)
      X = X.reshape(1, X.shape[0])
      X = torch.from_numpy(X).to(torch.float32)
      output = model(X)
      _, predicted = torch.max(output, dim=1)
      tag = tags[predicted.item()]
      probs = torch.softmax(output, dim=1)
      prob = probs[0][predicted.item()]
  
      if prob.item() >= 0.81:
          for intent in intents['intents']:
              if tag == intent["tag"]:
                  return random.choice(intent['responses'])
      else:
          return "I'm not sure how to respond to that. Can you please rephrase?"
  `,
        },
        {
          title: "Defining Routes for Web Pages and Message Sending",
          code: `@app.route('/')
  def home():
      return render_template('demoo.html')
  
  @app.route('/redirect/<page_name>')
  def redirect_to_page(page_name):
      return render_template(f'{page_name}.html')
  
  @app.route('/send-message', methods=['POST'])
  def send_message():
      message = request.json['message']
      response = process_input(message)
      return jsonify({'response': response})
  `,
        },
        {
          title: "Running the Flask App",
          code: `if __name__ == '__main__':
      app.run(debug=True)
  `,
        },
      ],
    },
];

  

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
          Codebase for the Workshop
        </motion.h1>
        <div className="space-y-8">
          {codeSnippets.map((file, fileIndex) => (
            <motion.div key={file.file} initial={{ opacity: 0, y: 20 }} animate={controls} custom={fileIndex}>
              <h2 className="text-2xl font-semibold text-accent mb-4">{file.file}</h2>
              <div className="space-y-4">
                {file.snippets.map((snippet, snippetIndex) => (
                  <motion.div key={snippet.title} initial={{ opacity: 0, y: 10 }} animate={controls} custom={snippetIndex}>
                    <CodeCard title={snippet.title} code={snippet.code} />
                  </motion.div>
                ))}
              </div>
            </motion.div>
          ))}
        </div>
      </main>
    </div>
  );
  
}

