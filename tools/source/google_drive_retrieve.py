def main(query: str) -> str:
    """
    Retrieve files from Google Drive based on a query.

    :param query: str: String to find relevant documents for
    :returns: str: Relevant documents
    """
    from langchain_googledrive.retrievers import GoogleDriveRetriever

    retriever = GoogleDriveRetriever(
        folder_id=Configs.google_drive_folder,
        template="gdrive-query",
        num_results=2,
        gdrive_token_path=Configs.gdrive_token_path,
    )

    # NOTE: this actually returns list[Document]
    # But I am matching from SecGPT's implementation
    return retriever.get_relevant_documents(query)
