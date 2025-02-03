"use client"

import { useEffect } from "react"
import CodeCard from "@/components/CodeCard"
import { motion, useAnimation } from "framer-motion"

export default function Home() {
  const controls = useAnimation()
  const codeSnippets = [
    {
      file: "Requirements.txt",
      snippets: [
        {
          title: "Requirements",
          code: `pip install flask nltk torch numpy pandas scikit-learn`,
        },
      ],
    },
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
    {
      file: "NeuralNetworks.py",
      snippets: [
        {
          title: "Perceptron",
          code: `import numpy as np

# sigmoid function to normalize inputs
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid derivatives to adjust synaptic weights
def sigmoid_derivative(x):
    return x * (1 - x)

# ipdataset
training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

# opdataset
training_outputs = np.array([[0,1,1,0]]).T

# seed random numbers to make calculation
np.random.seed(1)

# initialize weights randomly with mean 0 to create weight matrix, synaptic weights
synaptic_weights = 2 *  np.random.random((3,1)) - 1

print('Random starting synaptic weights: ')
print(synaptic_weights)

# Iterate 10,000 times
for iteration in range(1000): # x * wt -> y

    # Define input layer
    input_layer = training_inputs
    # Normalize the product of the input layer with the synaptic weights
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    # how much did we miss?
    error = training_outputs - outputs

    # multiply how much we missed by the
    # slope of the sigmoid at the values in outputs
    adjustments = error * sigmoid_derivative(outputs)

    # update weights
    synaptic_weights += np.dot(input_layer.T, adjustments)



print('Synaptic weights after training: ')
print(synaptic_weights)


print("Output After Training:")
print(outputs)


    

`,
        },
      ],
    },
    {
      file: "Data Inputs",
      snippets: [
        {
          title: "intents.json",
          code: `{
"intents": [
{
  "tag": "greeting",
  "patterns": [
    "Hi",
    "Hey",
    "Hello",
    "Good morning",
    "Good afternoon",
    "Greetings",
    "Howdy",
    "Hi there",
    "Hey there",
    "Morning",
    "Afternoon",
    "Yo",
    "What's up"
  ],
  "responses": [
    "Hi there! Welcome to TechSolutions. How may I assist you today?",
    "Hello! It's great to have you here. How can TechSolutions help you?",
    "Good morning! How can we make your day better?",
    "Hey! Welcome to TechSolutions. How can we assist you?",
    "Hi! How may I help you today?",
    "Hello there! How can TechSolutions make your day easier?",
    "Greetings! What can we do for you today?",
    "Howdy! Welcome to TechSolutions. How can we support you?",
    "Well met! How may I assist you today?",
    "Salutations! How can TechSolutions serve you?"
  ]
},
{
  "tag": "employee_schedule",
  "patterns": ["What is my schedule?", "When do I work next?", "Tell me my work hours", "Show me my schedule", "Do I have work tomorrow?", "What is my next shift?"],
  "responses": ["Fetching your schedule. Please wait a moment..."],
  "action": "fetch_employee_schedule"
},
{
  "tag": "leave_balance",
  "patterns": ["What is my leave balance?", "How many leave days do I have left?", "Tell me my remaining leave", "How many vacation days do I have?", "What is my current leave balance?"],
  "responses": ["Fetching your leave balance. Please wait a moment..."],
  "action": "fetch_leave_balance"
},
{
  "tag": "goodbye",
  "patterns": ["Bye", "See you later", "Goodbye", "Take care", "Catch you later", "Farewell", "Bye for now", "Have a good one", "Until next time", "Adios", "So long", "Later"],
  "responses": [
    "Goodbye! Don't hesitate to reach out if you need anything else.",
    "See you later! Have a fantastic day.",
    "Take care! We're looking forward to assisting you again soon.",
    "Bye! Have a great day ahead.",
    "Goodbye! Remember, we're just a message away if you need help.",
    "Farewell! Until next time.",
    "Take care! Don't hesitate to contact us if you have any questions.",
    "Bye for now! We'll be here whenever you need assistance."
  ]
},
{
  "tag": "thanks",
  "patterns": ["Thanks", "Thank you", "Appreciate it","Ok sure" ,"Thanks a lot", "Much obliged", "Cheers", "You're awesome", "Gracias", "Merci", "Danke", "Thank you very much"],
  "responses": [
    "You're welcome! Let us know if there's anything else we can assist you with.",
    "Anytime! We're here to help.",
    "You're welcome! It's our pleasure to assist you.",
    "Glad we could help! Feel free to ask if you need anything else.",
    "You're welcome! If you have any more questions, just let us know.",
    "No problem at all! We're happy to be of service.",
    "You're welcome! Don't hesitate to reach out if you need further assistance.",
    "Happy to assist! If you have any more questions, feel free to ask."
  ]
},
{
  "tag": "services",
  "patterns": [
    "What services do you offer?",
    "Can you help with IT support?",
    "What kind of solutions do you provide?",
    "what kind of services you provide",
    "Tell me about your services",
    "What can you do for me?",
    "What are the services you offer",
    "How do you help businesses?",
    "What do you specialize in?",
    "Do you offer tech solutions for small businesses?",
    "I need assistance with my IT infrastructure",
    "Do you provide cloud solutions?",
    "I'm looking for cybersecurity services",
    "Can you help with software development?",
    "Do you offer network setup services?",
    "Can you assist with data management?",
    "services"
    
  ],
  "responses": [
    "TechSolutions offers a comprehensive suite of IT services, including network infrastructure setup, software development, cybersecurity solutions, cloud computing, and more.",
    "Our IT support services range from troubleshooting common issues to implementing complex system upgrades and optimizations.",
    "We specialize in providing tailored IT solutions to meet the unique needs of your business, from data management to custom software development.",
    "At TechSolutions, we offer a wide range of IT services designed to streamline your business operations and enhance efficiency.",
    "From IT consulting to managed services, TechSolutions has the expertise to meet your technology needs and drive your business forward.",
    "Our services include IT strategy development, software implementation, cloud migration, cybersecurity, and ongoing support to ensure your systems run smoothly.",
    "Whether you need help with network infrastructure, software development, or cybersecurity, TechSolutions has you covered with our comprehensive range of services."
  ],
  "questions": [
    "Sure, could you please provide more details about the specific IT service or solution you're interested in?",
    "Great! To better assist you, could you tell me about your current IT infrastructure and any challenges you're facing?",
    "Which aspect of our services are you most interested in? We can provide more details based on your specific needs.",
    "Is there a particular area of your business where you're looking to improve efficiency or streamline operations? Our services can be customized to address your unique requirements."
  ]
},
{
  "tag": "contact",
  "patterns": [
    "How can I contact you",
    "What's your phone number?",
    "Do you have an email address?",
    "give me contact details",
    "contact information",
    "contact",
    "I need to get in touch with you",
    "Can I reach you by phone?",
    "What's the best way to reach you?",
    "Is there a contact form on your website?"
  ],
  "responses": [
    "You can reach TechSolutions by phone at +1 (555) 123-4567 or via email at info@techsolutions.com.",
    "Feel free to contact us anytime at +1 (555) 123-4567 or via email at info@techsolutions.com.",
    "For inquiries or assistance, please contact us at +1 (555) 123-4567 or email us at info@techsolutions.com.",
    "Our contact information is as follows: Phone: +1 (555) 123-4567, Email: info@techsolutions.com. Don't hesitate to reach out if you have any questions or need assistance."
  ],
  "questions": [
    "Of course! How can I assist you further? Are you looking to schedule a consultation or inquire about our services?",
    "Certainly! Is there a particular department or individual you're trying to reach, or do you have a general inquiry?",
    "Would you prefer to contact us by phone or email? Let me know how you'd like to proceed.",
    "Are you facing any specific challenges or issues that you'd like assistance with? Our team is here to help, so feel free to reach out."
  ]
},
{
  "tag": "pricing",
  "patterns": [
    "How much do your services cost?",
    "pricing of your services",
    "What are your rates?",
    "Can you provide a quote?",
    "I'm interested in your prices",
    "What's your pricing model?",
    "Do you offer free consultations?",
    "Are there any hidden fees?",
    "What payment methods do you accept?",
    "Are your prices negotiable?",
    "Are there any discounts available?",
    "Do you offer service packages?",
    "price",
    "cost"
  ],
  "responses": [
    "Our pricing is customized based on your specific requirements. Please contact our sales team for a personalized quote.",
    "We offer competitive rates tailored to your needs. Get in touch with us to discuss your project and receive a detailed pricing estimate.",
    "For pricing information, please reach out to our sales department. They'll be happy to provide you with a quote based on your project specifications.",
    "Since our services are tailored to each client's unique needs, pricing varies. We'd be happy to provide you with a personalized quote based on your requirements.",
    "At TechSolutions, we believe in transparent pricing. Contact us today to discuss your project, and we'll provide you with a detailed quote tailored to your needs."
  ],
  "questions": [
    "Certainly! To provide you with an accurate quote, could you please provide more details about your project requirements?",
    "Absolutely! Could you tell me more about the scope of your project and any specific features or functionalities you're looking to implement?",
    "Are there any budget constraints or preferences we should be aware of while preparing your quote?",
    "Do you have a timeframe for your project? Knowing your deadlines will help us provide you with an accurate quote."
  ]
},
{
  "tag": "hours",
  "patterns": [
    "What are the business hours?",
    "When are you open?",
    "What time can I reach you?",
    "When is your office open?",
    "Are you available on weekends?",
    "What are your support hours?",
    "When can I schedule a consultation?",
    "Do you offer evening appointments?",
    "Are you closed on holidays?",
    "What are your operating hours during the holidays?",
    "Can I visit your office without an appointment?"
  ],
  "responses": [
    "TechSolutions is open Monday to Friday from 9:00 AM to 6:00 PM.",
    "Our business hours are Monday through Friday, 9:00 AM to 6:00 PM. We're closed on weekends and major holidays.",
    "You can reach us during our business hours, which are Monday to Friday, 9:00 AM to 6:00 PM.",
    "Our office hours are from 9:00 AM to 6:00 PM, Monday through Friday. We're here to assist you during these times."
  ],
  "questions": [
    "Got it! Is there a specific time you're planning to visit our office, or do you have a general inquiry?",
    "Sure thing! Are you looking to schedule a meeting during our regular business hours, or do you have another inquiry?",
    "If you need assistance outside of our regular business hours, feel free to let us know, and we'll do our best to accommodate your request.",
    "Are you located in a different time zone? Let us know, and we'll adjust our availability accordingly to better serve you."
  ]
},
{
  "tag": "emergency",
  "patterns": [
    "Do you offer emergency support?",
    "What if I have an urgent issue?",
    "I have a critical problem",
    "My issue needs immediate attention",
    "Emergency assistance required",
    "I need help urgently",
    "Can you help me right away?",
    "My system is down",
    "I can't access my data",
    "Urgent IT problem",
    "I'm facing a cybersecurity threat",
    "My network is compromised"
  ],
  "responses": [
    "Yes, we provide 24/7 emergency support for critical issues. Please call our emergency hotline at +1 (555) 987-6543 for immediate assistance.",
    "In case of an urgent issue, don't hesitate to contact our emergency support team at +1 (555) 987-6543. We're here to help you around the clock.",
    "For urgent matters, you can rely on our emergency support team, available 24/7. Reach out to us at +1 (555) 987-6543, and we'll assist you promptly.",
    "We understand the importance of resolving critical issues promptly. Contact our emergency support hotline at +1 (555) 987-6543 for immediate assistance."
  ],
  "questions": [
    "Understood! Could you please provide more details about the nature of the emergency or issue you're experiencing?",
    "I see! To assist you more effectively, could you provide additional information about the urgency and impact of the issue?",
    "Is this a recurring issue, or is it something new that requires immediate attention? Providing more context will help us address the issue more efficiently.",
    "Are there any specific troubleshooting steps you've already taken? Let us know so we can better assist you."
  ]
},
{
  "tag": "employee_schedule",
  "patterns": ["what is the schedule for [employee_name]?", "When does [employee_name] work next?", "Tell me the work hours for [employee_name]", "Show me the schedule for [employee_name]", "Does [employee_name] have work tomorrow?", "What is [employee_name]'s next shift?"],
  "responses": ["Fetching the schedule for [employee_name]. Please wait a moment..."],
  "action": "fetch_employee_schedule"
},
{
  "tag": "leave_balance",
  "patterns": ["what is the leave balance for [employee_name]?", "How many leave days does [employee_name] have left?", "Tell me the remaining leave for [employee_name]", "How many vacation days does [employee_name] have?", "What is the current leave balance for [employee_name]?"],
  "responses": ["Fetching the leave balance for [employee_name]. Please wait a moment..."],
  "action": "fetch_leave_balance"
}
]
}`,
        },
        {
          title: "scripts.js",
          code: `const recognition = new webkitSpeechRecognition();
recognition.continuous = false;
recognition.lang = 'en-US';
recognition.interimResults = false;

const synthesis = window.speechSynthesis;
let isListening = false;

const startListeningButton = document.getElementById("start-listening-button");

document.getElementById("user-input").addEventListener("keypress", function(event) {
if (event.key === "Enter") {
    sendMessage();
}
});

function speak(text) {
var utterance = new SpeechSynthesisUtterance();
utterance.text = text;
synthesis.speak(utterance);
}

function showNotification() {
const chatContainer = document.querySelector(".chat-container");
const notification = document.createElement("div");
notification.textContent = "Microphone is on";
notification.classList.add("notification");
chatContainer.appendChild(notification);

notification.offsetHeight;
notification.classList.add("show"); 
setTimeout(() => {
    notification.classList.remove("show");
    setTimeout(() => {
        chatContainer.removeChild(notification);
    }, 500);
}, 3000);
}

function startListening() {
recognition.start();
isListening = true;
showNotification();
}

recognition.onresult = function(event) {
var transcript = event.results[0][0].transcript;
appendMessage("You: " + transcript, "right-message");
fetch("/send-message", {
    method: "POST",
    body: JSON.stringify({ message: transcript, useQuadBot: useQuadBot }),
    headers: {
        "Content-Type": "application/json"
    }
})
.then(response => response.json())
.then(data => {
    if (data.response) {
        setTimeout(() => {
            appendMessage("Bot: " + data.response, "left-message");
            setTimeout(() => {
                speak(data.response);
            }, 500);
        }, 300);
    } else {
        setTimeout(() => {
            appendMessage("Bot: I'm sorry, I didn't understand that.", "left-message");
            setTimeout(() => {
                speak("I'm sorry, I didn't understand that.");
            }, 500);
        }, 300);
    }
})
.catch(error => {
    console.error('Error sending message:', error);
    setTimeout(() => {
        appendMessage("Bot: Oops! Something went wrong.", "left-message");
        setTimeout(() => {
            speak("Oops! Something went wrong.");
        }, 500);
    }, 300);
});
}

recognition.onerror = function(event) {
console.error('Speech recognition error:', event.error);
isListening = false;
}

function appendMessage(message, side) {
var chatBox = document.getElementById("chat-box");
var newMessage = document.createElement("div");
newMessage.textContent = message;
newMessage.classList.add("message", side);
chatBox.appendChild(newMessage);

setTimeout(() => {
    newMessage.style.opacity = "1";
    newMessage.style.transform = "translateY(0)";
}, 100);

chatBox.scrollTop = chatBox.scrollHeight;
}

function sendMessage() {
var userInput = document.getElementById("user-input").value;
if (userInput.trim() !== "") {
    appendMessage("You: " + userInput, "right-message");
    document.getElementById("user-input").value = "";
    fetch("/send-message", {
        method: "POST",
        body: JSON.stringify({ message: userInput, useQuadBot: useQuadBot }),
        headers: {
            "Content-Type": "application/json"
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.response) {
            setTimeout(() => {
                appendMessage("Bot: " + data.response, "left-message");
                speak(data.response);
            }, 700);
        } else {
            setTimeout(() => {
                appendMessage("Bot: I'm sorry, I didn't understand that.", "left-message");
                speak("I'm sorry, I didn't understand that.");
            }, 400);
        }
    })
    .catch(error => {
        console.error('Error sending message:', error);
        setTimeout(() => {
            appendMessage("Bot: Oops! Something went wrong.", "left-message");
        }, 400);
    });
}
}

function toggleChat() {
const chatContainer = document.querySelector(".chat-container");
const chatButton = document.querySelector(".chat-button");

if (chatContainer.classList.contains("active")) {
    setTimeout(() => {
        chatContainer.classList.remove("active");
    }, 30);
} else {
    chatContainer.classList.add("active");
    const chatButtonRect = chatButton.getBoundingClientRect();
    chatContainer.style.top =  "260px";
    chatContainer.style.bottom = "1000px";
}
}

function showSection(sectionId) {
document.querySelectorAll('.content').forEach(section => {
    section.style.display = 'none';
});

const sectionToShow = document.getElementById(sectionId);
if (sectionToShow) {
    sectionToShow.style.display = 'block';
}
}

document.addEventListener('DOMContentLoaded', () => {
showSection('home');
});

let useQuadBot = true;

function switchBot() {
useQuadBot = !useQuadBot;
const botName = useQuadBot ? 'QuadBot' : 'Z Bot';
document.querySelector('.chat-container h2').textContent = botName;

}`,
        },
        {
          title: "styles.css",
          code: `/* Chatbot Styles */
.chat-container {
display: none;
flex-direction: column;
position: fixed;
bottom: 20px;
right: 20px;
max-width: 400px;
width: 100%;
height: 500px;
background-color: rgba(255, 255, 255, 0.9);
border-radius: 20px;
padding: 15px;
box-shadow: 0 0 15px rgba(0, 0, 0, 0.15);
border: 2px solid #0078d7;
z-index: 1000;
transition: all 0.3s ease; /* Add transition effect for chat container */
}

.chat-container #chat-box {
width: 100%;
height: calc(100% - 110px);
overflow-y: auto;
padding: 10px;
box-sizing: border-box;
background-color: rgba(241, 241, 241, 0.9);
border-radius: 10px;
}

.chat-container .message {
max-width: 70%;
margin-bottom: 10px;
padding: 10px;
border-radius: 10px;
line-height: 1.5;
font-size: 1em;
word-wrap: break-word;
position: relative;
opacity: 0; /* Start hidden */
transform: translateY(10px); /* Start with a slight downward position */
transition: opacity 0.5s ease, transform 0.5s ease; /* Smooth transition */
}

.chat-container .right-message {
background-color: rgba(0, 120, 215, 0.9);
color: #ffffff;
align-self: flex-end;
margin-left: auto;
border: 1px solid rgba(0, 91, 181, 0.9);
}

.chat-container .left-message {
background-color: rgba(52, 168, 83, 0.9);
color: #ffffff;
align-self: flex-start;
margin-right: auto;
border: 1px solid rgba(42, 139, 67, 0.9);
}

.chat-container button {
padding: 10px 15px;
border-radius: 5px;
margin-right: 5px;
cursor: pointer;
font-size: 1em;
transition: background-color 0.3s ease;
color: #ffffff;
border: none;
display: flex;
align-items: center;
justify-content: center;
}

.chat-container #start-btn {
background-color: rgba(255, 87, 34, 0.9);
}

.chat-container #send {
background-color: rgba(0, 120, 215, 0.9);
}

.chat-container button:hover {
opacity: 0.8;
}

.chat-container #user-input {
width: calc(100% - 100px);
border: none;
padding: 16px 13px;
border-radius: 10px;
font-size: 1.4em; 
outline: none;
}

.chat-container .button-container {
display: flex;
justify-content: space-between;
align-items: center;
margin-top: 10px;
}

.chat-container .notification {
background-color: rgba(255, 152, 0, 0.9);
color: #ffffff;
padding: 15px;
border-radius: 10px;
position: absolute;
top: 90px;
right: 20px;
font-size: 1.2em;
opacity: 0;
transition: opacity 0.5s ease-out;
}

.chat-container .notification.show {
opacity: 1;
}

/* Chatbot Button */
.chat-button {
position: fixed;
bottom: 20px;
right: 20px;
background-color: rgba(29, 60, 85, 0.9);
color: #ffffff;
border-radius: 50%;
padding: 15px;
box-shadow: 0 0 15px rgba(0, 0, 0, 0.15);
cursor: pointer;
z-index: 1000;
}

.chat-button i {
font-size: 24px;
}

/* Font Awesome Icons */
.chat-container button i {
margin: 0;
font-size: 36px;
}

.chat-container {
margin-bottom: 10px;
transition: all 0.3s ease; /* Add transition effect */
}

.chat-container.active {
display: flex;
}

/* Switch Bot Button */
.chat-container #switch-bot {
background-color: rgba(0, 153, 255, 0.9);
}

.chat-container #switch-bot:hover {
background-color: rgba(0, 120, 255, 0.9);
}

/* Existing CSS */
* {
margin: 0;
padding: 0;
font-family: "Poetsen One", sans-serif;
}

.header {
min-height: 100vh;
width: 100%;
background-image: linear-gradient(rgba(4, 9, 30, 0.7), rgba(4, 9, 30, 0.7)), url(ba.jpg.jpg);
background-size: cover;
background-position: center;
position: relative;
}

nav {
display: flex;
padding: 2% 6%;
justify-content: space-between;
align-items: center;
}

nav img {
width: 150px;
}

.navlink {
flex: 1;
text-align: right;
}

.navlink ul li {
list-style-type: none;
display: inline-block;
padding: 8px 12px;
position: relative;
}

.navlink ul li a {
color: aliceblue;
text-decoration: none;
font-size: 13px;
}

.navlink ul li::after {
content: '';
width: 0%;
height: 2px;
background: #f44336;
display: block;
margin: auto;
}

.navlink ul li:hover::after {
width: 100%;
transition: 0.5s;
}

.text-box {
width: 90%;
color: #fff;
position: absolute;
top: 50%;
left: 50%;
transform: translate(-50%, -50%);
text-align: center; /* Added to center text elements */
display: flex;
flex-direction: column;
align-items: center; /* Added to center child elements horizontally */
}

.text-box h1 {
font-size: 62px;
}

.text-box p {
margin: 10px 0 40px;
font-size: 14px;
color: white;
}

.hero-btn {
display: inline-block;
text-decoration: none;
color: white;
border: 1px solid #fff;
padding: 12px 34px;
font-size: 13px;
background: transparent;
position: relative;
cursor: pointer;
}
.hero-btn:hover {
border: 1px solid #11c942;
background: #59c5ec;
transition: 1s;
}

nav.fa {
display: none;
}

@media(max-width: 700px) {
.text-box h1 {
    font-size: 20px;
}

.navlink ul li {
    display: block;
}

.navlink {
    position: absolute;
    background: #f44336;
    height: 100vh;
    width: 200px;
    top: 0;
    right: -200px;
    text-align: left;
    z-index: 2;
    transition: 1s;
}

nav .fa {
    display: block;
    color: #fff;
    margin: 10px;
    font-size: 22px;
    cursor: pointer;
}

.navlink ul {
    padding: 30px;
}
}

/---- course-----/
.course {
width: 80%;
margin: auto;
text-align: center;
padding-top: 100px;
}

h1 {
font-size: 36px;
font-weight: 600;
}

p {
color: #777;
font-size: 15px;
font-weight: 300;
line-height: 22px;
padding: 10px;
}

.row {
margin-top: 3%;
display: flex;
justify-content: space-between;
}

.course-col {
flex-basis: 31%;
background: #fff3f3;
border-radius: 10px;
margin-bottom: 5%;
padding: 20px 12px;
box-sizing: border-box;
}

h3 {
text-align: center;
font-weight: 600;
margin: 10px 0;
}

.course-col:hover {
box-shadow: 0 0 20px 0px rgba(0, 0, 0, 0.2);
}

@media(max-width: 700px) {
.row {
    flex-direction: column;
}
}

.campus {
width: 80%;
margin: auto;
padding-top: 50px;
text-align: center;
}

.campus .row {
justify-content: space-between;
flex-wrap: wrap;
}

.campus-col {
border-radius: 10px;
margin-bottom: 30px;
position: relative;
overflow: hidden;
}

.campus-col img {
width: 65%;
transition: transform 0.5s; /* Add transition for smooth effect */
}

.campus-col:hover img {
transform: scale(1.1); /* Slightly enlarge the image on hover */
}

.campus-col .overlay {
position: absolute;
top: 0;
left: 0;
width: 100%;
height: 100%;
background: rgba(255, 0, 0, 0.5); /* Red overlay with some transparency */
opacity: 0;
transition: opacity 0.5s;
}

.campus-col:hover .overlay {
opacity: 1; /* Show the overlay on hover */
}

.layer h3 {
width: 100%;
font-weight: 500;
color: #fff;
font-size: 26px;
bottom: 0;
left: 50%;
transform: translate(-50%);
position: absolute;
}

.facilities {
width: 80%;
margin: auto;
text-align: center;
padding-top: 100px;
}

.facility-col {
flex-basis: 31%;
border-radius: 10px;
margin-bottom: 5%;
text-align: left;
}

.facility-col img {
width: 100%;
border-radius: 30%;
}

.facility-col h3 {
margin-top: 16px;
margin-bottom: 15px;
text-align: center;
}

.testimonials {
width: 80%;
margin: auto;
padding: top 100px;
text-align: center;
}

.testi-col {
flex-basis: 44%;
border-radius: 10px;
margin-bottom: 5%;
text-align: left;
background: #fff3f3;
padding: 25px;
cursor: pointer;
display: flex;
}

.testi-col img {
height: 40px;
margin-left: 5px;
margin-right: 30px;
border-radius: 50%;
}

.testi-col p {
padding: 0;
}

.testi-col h3 {
margin-top: 15px;
text-align: left;
}

@media (min-width: 701px) {
nav .fas.fa-bars,
nav .fas.fa-times {
    display: none;
}
}`,
        },
        {
          title: "demoo.html",
          code: `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TechSolutions</title>
<link href='https://unpkg.com/boxicons@2.1.4/css/boxicons.min.css' rel='stylesheet'>
<link rel="stylesheet" href="{{url_for('static', filename='styles.css')}}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Poetsen+One&family=Roboto:ital,wght@0,100;0,300;0,400;0,500;0,700;0,900;1,100;1,300;1,400;1,500;1,700;1,900&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.2/css/all.min.css"> <!-- Use all.min.css for FA 6 -->
</head>
<body>
<section class="header">
    <nav>
        <a href="/redirect/demoo"><img src="{{url_for('static', filename='neweagle.png')}}" alt="Logo"></a>
        <div class="navlink" id="navLink">
            <i class="fas fa-times" onclick="hidemenu()"></i>
            <ul>
                <li><a href="/redirect/demoo">HOME</a></li>
                <li><a href="/redirect/about">ABOUT</a></li>
                <li><a href="/redirect/blog">BLOG</a></li>
                <li><a href="/redirect/contact">CONTACT</a></li>
            </ul>
        </div>
        <i class="fas fa-bars" onclick="showmenu()"></i>
    </nav>
    <div class="text-box">
        <h1>WORLD'S BIGGEST TECH-SERVICES</h1>
        <p>We innovate to simplify your life with cutting-edge technology,<br> delivering solutions that empower and inspire.</p>
        <a href="https://www.amazon.in/ref=nav_logo" class="hero-btn">Visit us to know more</a>
    </div>
</section>

<section class="course">
    <h1>Create a Module From a Template</h1>
    <p>Lorem ipsum dolor sit amet consectetur adipisicing elit. Pariatur architecto, veritatis porro cumque sint mollitia totam alias at error? Eum, atque repudiandae ratione dolore mollitia facere libero assumenda sint tempore.</p>
    <p>The second way to create a module in Slider Revolution is from a template. Let's learn why, and how, you should make a template-based module.</p>
    <div class="row">
        <div class="course-col">
            <h3>Creating Module</h3>
            <p>Everything covered in this ‘Create a Module from a Template’ sub-section of the manual can also be learned by watching the tutorial video below from the 6:23 minute mark:</p>
        </div>
        <div class="course-col">
            <h3>Why Use a Template</h3>
            <p>You want to create content quickly by leveraging a template to give you a head start.<br>You found something you love in the library of templates included with Slider Revolution and want to use it on your site.</p>
        </div>
        <div class="course-col">
            <h3>Browsing Template</h3>
            <p>In the main Slider Revolution interface, the button to create a module from a template is labeled New Module From Template.</p>
        </div>
    </div>
</section>

<section class="campus">
    <h1>Our company</h1>
    <div class="campus-col">
        <img src="{{url_for('static', filename='company1.jpg.webp')}}" alt="Mumbai">
        <div class="cc">
            <h3>MUMBAI</h3>
        </div>
    </div>
    <div class="campus-col">
        <img src="{{url_for('static', filename='zoho.png')}}" alt="Delhi">
        <div class="cc">
            <h3>DELHI</h3>
        </div>
    </div>
    <div class="campus-col">
        <img src="{{url_for('static', filename='company2.jpg.webp')}}" alt="Chennai">
        <div class="cc">
            <h3>CHENNAI</h3>
        </div>
    </div>
</section>

<section class="facilities">
    <h1>OUR FACILITIES</h1>
    <p>Lorem ipsum, dolor sit amet consectetur adipisicing elit. Sapiente placeat ipsam nihil expedita fugiat quae non quam dolorum? Animi dignissimos eveniet quam modi esse omnis nihil id sed, quaerat dolorum.</p>
    <div class="facility-col">
        <img src="{{url_for('static', filename='lib1.jpg')}}" alt="Library">
        <h3>World class library</h3>
    </div>
    <div class="facility-col">
        <img src="{{url_for('static', filename='baby1.png')}}" alt="Baby Sitting Room">
        <h3>A secured baby sitting room</h3>
    </div>
    <div class="facility-col">
        <img src="{{url_for('static', filename='sports1.png')}}" alt="Sports Room">
        <h3>World class sportsroom</h3>
    </div>
</section>

<section class="testimonials">
    <h1>What our clients says</h1>
    <div class="testi-col">
        <img src="{{url_for('static', filename='clients1.jpg')}}">

        <div>
            <p>Lorem ipsum dolor sit amet consectetur adipisicing elit. </p>
            <h3>BHRAHMESH</h3>
            <i class="fa fa-star"></i>
            <i class="fa fa-star"></i>
            <i class="fa fa-star"></i>
            <i class="fa fa-star"></i>
            <i class="fa fa-star-half"></i>
        </div>
    </div>
    <div class="testi-col">
        <img src="{{url_for('static', filename='clients2.jpg')}}">

        <div>
            <p>Lorem ipsum dolor sit amet consectetur adipisicing elit. </p>
            <h3>KRISHNA</h3>
            <i class="fa fa-star"></i>
            <i class="fa fa-star"></i>
            <i class="fa fa-star"></i>
            <i class="fa fa-star"></i>
            <i class="fa fa-star"></i>
        </div>
            </div>
            </div>
    <div class="testi-col">
    <img src="{{url_for('static', filename='clients3.webp')}}">
    <div>
        <p>Lorem ipsum dolor sit amet consectetur adipisicing elit. </p>
        <h3>MATHEW</h3>
        <i class="fa fa-star"></i>
        <i class="fa fa-star"></i>
        <i class="fa fa-star"></i>
        <i class="fa fa-star"></i>
        <i class="fa fa-star"></i>
    </div>
</div>

</section>

<div class="chat-button" onclick="toggleChat()">
<i class='bx bx-chat bx-lg'></i>
</div>
<div class="chat-container" id="chat-container">
<h2 style="text-align: center; color: #312e2e; margin-bottom: 15px;">QuadBot</h2>
<div id="chat-box"></div>
<div class="button-container">
    <input type="text" id="user-input" placeholder="Type a message...">
    <button id="start-btn" onclick="startListening()"><i class='bx bx-microphone bx-md'></i></button>
    <button id="send" onclick="sendMessage()"><i class='bx bx-send bx-md'></i></button>
    <button id="switch-bot" onclick="switchBot()"><i class='bx bx-transfer-alt'></i></button> <!-- Bot switch button -->
</div>
</div>
<script src="{{url_for('static', filename='script.js')}}"></script>
<script>
var navLink = document.getElementById("navLink");
function showmenu(){
    navLink.style.right = "0";
}
function hidemenu(){
    navLink.style.right = "-200px";
}
</script>
</body>
</html>`,
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

