import argparse
import csv
import time
import numpy as np
import joblib
import modelP

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
from threading import Thread
from typing import Union, List, Optional
from pathlib import Path
from pylsl import StreamInlet, resolve_byprop
from sklearn.linear_model import LinearRegression
from time import time, strftime, gmtime
from your_module.stream import find_muse
from your_module import backends
from your_module.muse import Muse
from your_module.constants import LSL_SCAN_TIMEOUT, LSL_EEG_CHUNK, LSL_PPG_CHUNK, LSL_ACC_CHUNK, LSL_GYRO_CHUNK

# Global data buffer to store incoming data
data_buffer = []

def record(
    duration: int,
    filename=None,
    dejitter=False,
    data_source="EEG",
    continuous: bool = True,
) -> None:
    chunk_length = LSL_EEG_CHUNK
    if data_source == "PPG":
        chunk_length = LSL_PPG_CHUNK
    if data_source == "ACC":
        chunk_length = LSL_ACC_CHUNK
    if data_source == "GYRO":
        chunk_length = LSL_GYRO_CHUNK

    if not filename:
        filename = os.path.join(os.getcwd(), "%s_recording_%s.csv" %
                                (data_source,
                                 strftime('%Y-%m-%d-%H.%M.%S', gmtime())))

    print("Looking for a %s stream..." % (data_source))
    streams = resolve_byprop('type', data_source, timeout=LSL_SCAN_TIMEOUT)

    if len(streams) == 0:
        print("Can't find %s stream." % (data_source))
        return

    print("Started acquiring data.")
    inlet = StreamInlet(streams[0], max_chunklen=chunk_length)
    # eeg_time_correction = inlet.time_correction()

    print("Looking for a Markers stream...")
    marker_streams = resolve_byprop(
        'name', 'Markers', timeout=LSL_SCAN_TIMEOUT)

    if marker_streams:
        inlet_marker = StreamInlet(marker_streams[0])
    else:
        inlet_marker = False
        print("Can't find Markers stream.")

    info = inlet.info()
    description = info.desc()

    Nchan = info.channel_count()

    ch = description.child('channels').first_child()
    ch_names = [ch.child_value('label')]
    for i in range(1, Nchan):
        ch = ch.next_sibling()
        ch_names.append(ch.child_value('label'))

    res = []
    timestamps = []
    markers = []
    t_init = time()
    time_correction = inlet.time_correction()
    last_written_timestamp = None
    print('Start recording at time t=%.3f' % t_init)
    print('Time correction: ', time_correction)
    while (time() - t_init) < duration:
        try:
            data, timestamp = inlet.pull_chunk(
                timeout=1.0, max_samples=chunk_length)

            if timestamp:
                res.append(data)
                timestamps.extend(timestamp)
                tr = time()
            if inlet_marker:
                marker, timestamp = inlet_marker.pull_sample(timeout=0.0)
                if timestamp:
                    markers.append([marker, timestamp])

            # Save every 5s
            if continuous and (last_written_timestamp is None or last_written_timestamp + 5 < timestamps[-1]):
                _save(
                    filename,
                    res,
                    timestamps,
                    time_correction,
                    dejitter,
                    inlet_marker,
                    markers,
                    ch_names,
                    last_written_timestamp=last_written_timestamp,
                )
                last_written_timestamp = timestamps[-1]

        except KeyboardInterrupt:
            break

    time_correction = inlet.time_correction()
    print("Time correction: ", time_correction)

    _save(
        filename,
        res,
        timestamps,
        time_correction,
        dejitter,
        inlet_marker,
        markers,
        ch_names,
    )

    print("Done - wrote file: {}".format(filename))


def _save(
    filename: Union[str, Path],
    res: list,
    timestamps: list,
    time_correction,
    dejitter: bool,
    inlet_marker,
    markers,
    ch_names: List[str],
    last_written_timestamp: Optional[float] = None,
):
    res = np.concatenate(res, axis=0)
    timestamps = np.array(timestamps) + time_correction

    if dejitter:
        y = timestamps
        X = np.atleast_2d(np.arange(0, len(y))).T
        lr = LinearRegression()
        lr.fit(X, y)
        timestamps = lr.predict(X)

    res = np.c_[timestamps, res]
    data = pd.DataFrame(data=res, columns=["timestamps"] + ch_names)

    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if inlet_marker and markers:
        n_markers = len(markers[0][0])
        for ii in range(n_markers):
            data['Marker%d' % ii] = 0
        # process markers:
        for marker in markers:
            # find index of markers
            ix = np.argmin(np.abs(marker[1] - timestamps))
            for ii in range(n_markers):
                data.loc[ix, "Marker%d" % ii] = marker[0][ii]

    # If file doesn't exist, create with headers
    # If it does exist, just append new rows
    if not Path(filename).exists():
        # print("Saving whole file")
        data.to_csv(filename, float_format='%.3f', index=False)
    else:
        # print("Appending file")
        # truncate already written timestamps
        data = data[data['timestamps'] > last_written_timestamp]
        data.to_csv(filename, float_format='%.3f', index=False, mode='a', header=False)

# Handler function to process incoming OSC messages
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

# Periodically save data to CSV
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
