"""
Entry point for the reinforcement learning training process.
"""

from flight_manger import get_client, start_training


def main():
    """
    Entry point for the reinforcement learning training process.

    Retrieves a kRPC client and passes it to the `start_training` function to
    begin the training process.
    """
    client = get_client()
    start_training(client=client)


if __name__ == "__main__":
    main()
