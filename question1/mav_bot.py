import streamlit as st
import random
import time
# from dotenv import load_dotenv
import torch
import pandas as pd
from torch import nn
import torch._dynamo
torch._dynamo.config.suppress_errors = True

block_size = 8

embedding_dim = 32

hidden_layer_size = 1024

activation = 'relu'

k = 10


st.set_page_config(page_title="Next K Words Predictor", page_icon="M")

# **Pinned Dropdowns on Sidebar**
with st.sidebar:
    st.write("### Model Configurations")
    
    block_size = st.selectbox(
        "Select Block Size:", options=[5,8]
    )
    embedding_dim = st.selectbox(
        "Select Embedding Dimensions:", options=[32, 64]
    )
    hidden_layer_size = st.selectbox(
        "Select Hidden Layer Size:", options=[512, 1024]
    )
    activation = st.selectbox(
        "Select Hidden Layer Size:", options=['relu', 'tanh']
    )
    k = st.text_input(
        "K:", value='5'
    )

    # Display the selected values
    # st.write(f"**Selected Block Size:** {block_size}")
    # st.write(f"**Selected Embedding Dimension:** {embedding_dim}")
    # st.write(f"**Selected Hidden Layer Size:** {hidden_layer_size}")

# **Chatbot Response Generator**


device = torch.device("cpu")

class NextChar(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size*emb_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)
        if activation=='relu':
            x = torch.relu(self.lin1(x))
        else:
            x = torch.tanh(self.lin1(x))
        x = self.lin2(x)
        return x 
    

model = NextChar(block_size, 11189, embedding_dim, hidden_layer_size)

# old_path = /Users/na/Machine Learning Assignment/Assignment-3/models_notebooks/models/model_{block_size}_{embedding_dim}_{hidden_layer_size}_{activation[0]}.pth
state_model = torch.load(f"question1/models_notebooks/models/model_{block_size}_{embedding_dim}_{hidden_layer_size}_{activation[0]}.pth", map_location=torch.device('cpu'))

new_state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_model.items()}

model.load_state_dict(new_state_dict)

i_s = pd.read_csv('/Users/na/Machine Learning Assignment/Assignment-3/itos.csv')
i = i_s['i']
s = i_s['s']
stoi = {sx:ix for ix,sx in zip(i,s)}
stoi[''] = 2
stoi["<S>"] = 0
stoi["."] = 1
itos = {i:s for s,i in stoi.items()}


g = torch.Generator()
g.manual_seed(40)
def generate_words(model, itos, stoi, block_size, max_len=100, context = [0] * block_size):
    name = ''
    for i in range(max_len):

        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        ch = itos[ix]
        if ch == '<S>':
            name += '.'
        elif ch not in ['"', ',', '.', '!', "'", "-", "?", "(", ")", ";", ":"]:
            name += ' '+ ch
        else:
            name += ch
        context = context[1:] + [ix]
    return name

def response_generator(inp):


    words = inp

    words = words.lower()
    words = words.strip()
    words = words.replace("\n", "")
    words = words.replace('"', '')
    words = words.replace(',', '')
    words = words.replace('.', '')
    words = words.replace('!', '')
    words = words.replace("'", "")
    words = words.replace("-", "")
    words = words.replace("?", "")
    words = words.replace("(", "")
    words = words.replace(")", "")
    words = words.replace(";", "")
    words = words.replace(":", "")
    words = words.strip()

    words = words.split(" ")
    if inp=='':
        words = "<S>"
    # print(words)
    for i in range(len(words)):
        if words[i] not in stoi.keys():
            print("Hello ")
            words[i] = "<S>"

    if len(words)>=block_size:
        words = words[-5:]
    else:
        words = ["<S>"]*(block_size-len(words)) + words
    context = [stoi[iw] for iw in words]
    response = (inp + " "+ generate_words(model, itos, stoi, block_size, max_len=int(k), context = context))
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


st.title('Next K Words Predictor')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    
    st.session_state.messages.append({"role": "assistant", "content": response})
