def main(message_id: str) -> str:
    """
    Delete a Gmail message by message_id.

    :param message_id: str: The ID of email message to be deleted.
    :returns: str: A message to indicate whether the message is successfully deleted.
    """
    return f"The email {message_id} has been deleted"
