import argparse
import csv
import time
import numpy as np
import joblib
import modelP

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
from threading import Thread

data_buffer = []

def print_petal_stream_handler(unused_addr, *args):
    sample_id = args[0]
    unix_ts = args[1] + args[2]
    lsl_ts = args[3] + args[4]
    data = args[5:]
    print(
        f'sample_id: {sample_id}, unix_ts: {unix_ts}, '
        f'lsl_ts: {lsl_ts}, data: {data}'
    )
    data_buffer.append((sample_id, unix_ts, lsl_ts, *data))

def save_data_periodically(duration):
    global data_buffer
    start_time = time.time()
    while time.time() - start_time < duration:
        time.sleep(2)  # Wait for 2 seconds
        # Save data to CSV file
        if data_buffer:
            with open('recorded_data.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(data_buffer)
            data_buffer = []  # Clear the buffer
    print("Data recording complete. Saved to 'recorded_data.csv'.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--ip', type=str, required=False,
                        default="127.0.0.1", help="The IP to listen on")
    parser.add_argument('-p', '--udp_port', type=str, required=False, default=14739,
                        help="The UDP port to listen on")
    parser.add_argument('-t', '--topic', type=str, required=False,
                        default='/PetalStream/eeg', help="The topic to print")
    parser.add_argument('-d', '--duration', type=int, required=False, default=20,
                        help="Duration to record data in seconds")
    args = parser.parse_args()

    dispatcher = Dispatcher()
    dispatcher.map(args.topic, print_petal_stream_handler)

    server = ThreadingOSCUDPServer(
        (args.ip, args.udp_port),
        dispatcher
    )

    print(f"Serving on {server.server_address}")

    # Start the server in a separate thread
    server_thread = Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    # Save data for the specified duration
    save_data_periodically(args.duration)

    # Stop the server after recording is done
    server.shutdown()

if __name__ == "__main__":
    main()
