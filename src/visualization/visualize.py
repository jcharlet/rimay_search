import streamlit as st
from src.models.predict_model import (
    run_query_with_qa_with_sources,
    COL_STATE_OF_THE_UNION,
    COL_OPEN_MINDFULNESS,
    ResponseSize,
)


# Define function to run query and display results
def run_query(
    query,
    mocked=False,
    collection_name=COL_OPEN_MINDFULNESS,
    response_size=ResponseSize.MEDIUM,
):
    if mocked:
        answer = """The president said "Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service." """
        sources = "[31-pl](https://www.openmindfulness.net/1-chapitre-2-e3/)"
        metadata = {
            "cost": {
                "Total Cost (USD)": "$0.05708",
                "Successful Requests": 2,
            },
            "tokens": {
                "Total Tokens": 2854,
                "Prompt Tokens": 2688,
                "Completion Tokens": 166,
            },
        }
    else:
        answer, sources, metadata = run_query_with_qa_with_sources(
            query, collection_name=collection_name, response_size=response_size
        )

    # = "Total Tokens: 2854 \nPrompt Tokens: 2688 \nCompletion Tokens: 166 \nSuccessful Requests: 2 \nTotal Cost (USD): $0.05708"
    return answer, sources, metadata


def make_grid(cols, rows):
    grid = [0] * cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid


# Define main function for Streamlit app
def main():
    st.title("OpenMindfulness Query App")

    language = ""
    token = ""
    st.sidebar.title("Settings")

    # # Add language selector to sidebar
    # language = st.sidebar.selectbox("Language", ["English", "French"])
    collection_name = st.sidebar.selectbox(
        "Collection name", [COL_OPEN_MINDFULNESS, COL_STATE_OF_THE_UNION]
    )
    is_mocked = st.sidebar.selectbox("is mocked", [True, False])
    response_size = st.sidebar.selectbox(
        "is mocked", [ResponseSize.SMALL, ResponseSize.MEDIUM, ResponseSize.LARGE]
    )

    # # Add text input for OpenAPI token to sidebar
    # token = st.sidebar.text_input("OpenAPI Token")

    # Add text input and button for user to enter query
    # query = st.text_input("Enter your query here", "Comment intégrer ses émotions avec la méthode en trois temps ?")
    query = st.text_input(
        "Enter your query here", "What did the president say about Justice Breyer ?"
    )

    if st.button("Run Query") and query != "":
        answer, sources, metadata = run_query(
            query,
            mocked=is_mocked,
            collection_name=collection_name,
            response_size=response_size,
        )
        st.header("Answer")
        st.markdown(answer)
        st.header("Sources")
        st.markdown(sources)
        st.header("Metadata")
        for k, v in metadata.items():
            # st.subheader(k)
            grid = make_grid(1, len(v.items()))
            for index, (k2, v2) in enumerate(v.items()):
                grid[0][index].metric(k2, v2)


# Run main function
if __name__ == "__main__":
    main()
