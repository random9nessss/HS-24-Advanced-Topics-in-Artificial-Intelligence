from speakeasypy import Speakeasy, Chatroom
from typing import List
import time
from agent_service import AgentService

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 2


class Agent:
    def __init__(self, username, password):
        self.username = username
        # Initialize the Speakeasy Python framework and login.
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.speakeasy.login()  # This framework will help you log out automatically when the program terminates.
        self.bot_service = AgentService()
        self.last_messages = {}

    def listen(self):
        while True:
            # only check active chatrooms (i.e., remaining_time > 0) if active=True.
            rooms: List[Chatroom] = self.speakeasy.get_rooms(active=True)
            for room in rooms:
                if not room.initiated:
                    # send a welcome message if room is not initiated
                    room.post_messages(f'Hello! This is a welcome message from {room.my_alias}.')
                    room.initiated = True
                # Retrieve messages from this chat room.
                # If only_partner=True, it filters out messages sent by the current bot.
                # If only_new=True, it filters out messages that have already been marked as processed.
                for message in room.get_messages(only_partner=True, only_new=True):
                    #print(
                        #f"\t- Chatroom {room.room_id} "
                        #f"- new message #{message.ordinal}: '{message.message}' "
                        #f"- {self.get_time()}")

                    ## ---------------------------------------------------- ##
                    # Implement your agent here #
                    ## ---------------------------------------------------- ##

                    if room.room_id not in self.last_messages:
                        self.last_messages[room.room_id] = []

                    last_exchange = self.last_messages[room.room_id][-1] if self.last_messages[room.room_id] else ("", "")
                    last_user_query, last_assistant_response = last_exchange

                    response_str = self.bot_service.react(message.message, last_user_query, last_assistant_response)
                    room.post_messages(f"{response_str}")

                    self.last_messages[room.room_id].append((message.message, response_str))
                    self.last_messages[room.room_id] = self.last_messages[room.room_id][-2:]

                    ## ---------------------------------------------------- ##
                    # Implement your agent here #
                    ## ---------------------------------------------------- ##


                    # Mark the message as processed, so it will be filtered out when retrieving new messages.
                    room.mark_as_processed(message)

                # Retrieve reactions from this chat room.
                # If only_new=True, it filters out reactions that have already been marked as processed.
                for reaction in room.get_reactions(only_new=True):
                    #print(
                        #f"\t- Chatroom {room.room_id} "
                        #f"- new reaction #{reaction.message_ordinal}: '{reaction.type}' "
                        #f"- {self.get_time()}")

                    room.post_messages(f"Received your reaction: '{reaction.type}' ")
                    room.mark_as_processed(reaction)

            time.sleep(listen_freq)

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())



if __name__ == '__main__':
    demo_bot = Agent("restless-bear", "I8qbH7J4")
    demo_bot.listen()
