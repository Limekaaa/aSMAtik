class Mailbox:
    def __init__(self):
        self._unread_messages = {}
        self._read_messages = {}
        self.history = []  # Optional: to keep a history of all messages sent

    def send_message(self, sender_id, recipient_ids, content):
        message = {
            "message_id": len(self.history),  # ID on history length for uniqueness
            "sender_id": sender_id,
            "recipient_ids": recipient_ids,
            "content": content,
        }
        self.history.append( message) # Keep history of sent messages
        for recipient_id in recipient_ids:
            if recipient_id not in self._unread_messages:
                self._unread_messages[recipient_id] = []
            self._unread_messages[recipient_id].append(message)

    def read_messages(self, recipient_id):
        if recipient_id not in self._unread_messages:
            return []
        
        messages = self._unread_messages.pop(recipient_id)

        if recipient_id not in self._read_messages:
            self._read_messages[recipient_id] = []

        self._read_messages[recipient_id].extend(messages)

        return messages
    
    def get_read_messages(self, recipient_id):
        return self._read_messages.get(recipient_id, [])
    
    def get_history(self):
        return self.history