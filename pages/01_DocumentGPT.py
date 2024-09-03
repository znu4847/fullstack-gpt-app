import time
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="🧊",
)

st.title("DocumentGPT")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

messages = st.session_state["messages"]

message = st.chat_input("Type something")


def send_message(message, role):
    with st.chat_message(role):
        st.write(message)
        messages.append({"message": message, "role": role})


if message:
    send_message(message, "human")
    time.sleep(2)
    send_message("How can I help you?", "ai")

    with st.sidebar:
        st.write(messages)


# with st.status("Embedding file...", expanded=True) as status:
#     time.sleep(2)
#     st.write("Getting the file")
#     time.sleep(2)
#     st.write("Embedding the file")
#     time.sleep(3)
#     st.write("Chashing the file")
#     status.update(label="Error", state="error")
