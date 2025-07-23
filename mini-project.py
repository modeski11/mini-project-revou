from typing_extensions import TypedDict
from agents.DOCSQNA import graph
from typing import List
import streamlit as st 
import uuid
from datetime import datetime

class Message(TypedDict):
    role: str
    content: str

class QnaState(TypedDict):
    messages: List[Message]

class Conversation(TypedDict):
    id: str
    title: str
    messages: List[Message]
    created_at: datetime

# Initialize session state
if 'conversations' not in st.session_state:
    st.session_state['conversations'] = []

if 'current_conversation_id' not in st.session_state:
    # Create initial conversation
    initial_conversation = Conversation(
        id=str(uuid.uuid4()),
        title="New Chat",
        messages=[{'role': 'assistant', 'content': 'Halo, adakah yang ingin anda ketahui tentang Dexa Medica?'}],
        created_at=datetime.now()
    )
    st.session_state['conversations'].append(initial_conversation)
    st.session_state['current_conversation_id'] = initial_conversation['id']

def get_current_conversation():
    """Get the current active conversation"""
    for conv in st.session_state['conversations']:
        if conv['id'] == st.session_state['current_conversation_id']:
            return conv
    return None

def create_new_conversation():
    """Create a new conversation and set it as current"""
    new_conversation = Conversation(
        id=str(uuid.uuid4()),
        title="New Chat",
        messages=[{'role': 'assistant', 'content': 'Halo, adakah yang ingin anda ketahui tentang Dexa Medica?'}],
        created_at=datetime.now()
    )
    st.session_state['conversations'].append(new_conversation)
    st.session_state['current_conversation_id'] = new_conversation['id']
    st.rerun()

def delete_conversation(conversation_id):
    """Delete a conversation"""
    st.session_state['conversations'] = [
        conv for conv in st.session_state['conversations'] 
        if conv['id'] != conversation_id
    ]
    
    # If we deleted the current conversation, switch to another or create new
    if st.session_state['current_conversation_id'] == conversation_id:
        if st.session_state['conversations']:
            st.session_state['current_conversation_id'] = st.session_state['conversations'][0]['id']
        else:
            create_new_conversation()
            return
    st.rerun()

def clear_all_history():
    """Clear all conversation history"""
    st.session_state['conversations'] = []
    create_new_conversation()

def update_conversation_title(conversation_id, new_title):
    """Update conversation title"""
    for conv in st.session_state['conversations']:
        if conv['id'] == conversation_id:
            conv['title'] = new_title
            break

def generate_title_from_first_message(messages):
    """Generate a title from the first user message"""
    for msg in messages:
        if msg['role'] == 'user':
            # Take first 30 characters and add "..." if longer
            title = msg['content'][:30]
            if len(msg['content']) > 30:
                title += "..."
            return title
    return "New Chat"

# Sidebar
with st.sidebar:
    st.title("💬 Chat History")
    
    # New Chat button
    if st.button("➕ New Chat", use_container_width=True):
        create_new_conversation()
    
    st.divider()
    
    # Conversation list
    current_conv = get_current_conversation()
    
    if st.session_state['conversations']:
        for i, conversation in enumerate(reversed(st.session_state['conversations'])):
            # Create container for each conversation
            conv_container = st.container()
            
            with conv_container:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Conversation button
                    is_current = conversation['id'] == st.session_state['current_conversation_id']
                    button_type = "primary" if is_current else "secondary"
                    
                    if st.button(
                        conversation['title'], 
                        key=f"conv_{conversation['id']}", 
                        use_container_width=True,
                        type=button_type
                    ):
                        st.session_state['current_conversation_id'] = conversation['id']
                        st.rerun()
                
                with col2:
                    # Delete button
                    if st.button("🗑️", key=f"del_{conversation['id']}", help="Delete conversation"):
                        delete_conversation(conversation['id'])
            
            # Show timestamp
            st.caption(f"Created: {conversation['created_at'].strftime('%m/%d %H:%M')}")
            st.divider()
    
    # Clear all history button
    if st.session_state['conversations']:
        st.markdown("---")
        if st.button("🗑️ Clear All History", use_container_width=True, type="secondary"):
            if st.button("Confirm Clear All", key="confirm_clear", type="primary"):
                clear_all_history()

# Main chat area
st.title("🤖 Dexa Medica Q&A Assistant")

# Get current conversation
current_conversation = get_current_conversation()

if current_conversation:
    # Display chat messages
    for message in current_conversation['messages']:
        with st.chat_message(message['role']):
            st.write(message['content'])

    # Chat input
    user_input = st.chat_input("Ketik pesan...")
    if user_input:
        # Add user message to current conversation
        current_conversation['messages'].append({'role': 'user', 'content': user_input})
        
        # Update conversation title if it's still "New Chat" and this is the first user message
        if current_conversation['title'] == "New Chat":
            new_title = generate_title_from_first_message(current_conversation['messages'])
            update_conversation_title(current_conversation['id'], new_title)
        
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            answer_placeholder = st.empty()
            status_placeholder.status(label="Process Start")
            state = "Process Start"
            prompt = QnaState(messages=current_conversation['messages'])
            final_answer = ""
            
            for chunk, metadata in graph.stream(prompt, stream_mode="messages"):
                if metadata['langgraph_node'] != state:
                    status_placeholder.status(label=metadata['langgraph_node'])
                    state = metadata['langgraph_node']
                    final_answer = "" 
                if metadata['langgraph_node'] == "respond":
                    final_answer += chunk.content
                    answer_placeholder.markdown(final_answer)
            
            status_placeholder.status(label="Complete", state='complete')
            
            # Add assistant response to current conversation
            current_conversation['messages'].append({'role': 'assistant', 'content': final_answer})

else:
    st.error("No conversation found. Please create a new chat.")
    if st.button("Create New Chat"):
        create_new_conversation()